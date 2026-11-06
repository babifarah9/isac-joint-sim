"""CP-OFDM point-target echo simulation and FFT range-Doppler estimator.
Research approximation: NOT 3GPP TS 38.211 pilot mapping, measured RF signatures,
a standards-conformant receiver, or an independently validated detector.
"""
from dataclasses import dataclass
import numpy as np
C = 299792458.0

@dataclass(frozen=True)
class WaveformConfig:
    frequency_ghz: float = 3.5
    channel_bandwidth_mhz: float = 80.0
    scs_khz: int = 30
    tx_power_dbm: float = 43.
    antenna_gain_dbi: float = 16.
    noise_figure_db: float = 7.
    sensing_snr_db_at_100m: float = 26.
    integration_ms: float = 20.
    sensing_share: float = .12
    traffic_load: float = .65
    ue_distance_m: float = 450.
    n_symbols: int = 128
    seed: int = 7

    def validate(self):
        if not .7 <= self.frequency_ghz <= 52.6: raise ValueError("Frequency out of range")
        if not 5 <= self.channel_bandwidth_mhz <= 400: raise ValueError("Bandwidth out of range")
        if self.scs_khz not in (15,30,60,120,240): raise ValueError("Unsupported subcarrier spacing")
        if not 5 <= self.integration_ms <= 1000: raise ValueError("Integration time out of range")
        if self.n_symbols not in (64,128,256): raise ValueError("Invalid sensing-symbol count")
        if not 0 <= self.sensing_share <= .6: raise ValueError("Invalid sensing allocation")
        if not 0 <= self.traffic_load <= 1: raise ValueError("Invalid traffic allocation")
        if not -20 <= self.sensing_snr_db_at_100m <= 60: raise ValueError("Invalid reference SNR")
        if not 0 <= self.tx_power_dbm <= 65: raise ValueError("Invalid transmitter power")
        if not 25 <= self.ue_distance_m <= 5000: raise ValueError("Invalid UE range")

@dataclass(frozen=True)
class Target:
    label: str
    range_m: float
    radial_velocity_mps: float
    rcs_dbsm: float = 0.
    category: str = "Unclassified (scenario truth only)"

    def validate(self):
        if not 5 <= self.range_m <= 5000: raise ValueError("Target range out of bounds")
        if not -100 <= self.radial_velocity_mps <= 100: raise ValueError("Target velocity out of bounds")

def simulate_ofdm(cfg: WaveformConfig, targets: list[Target], frame: int = 0):
    cfg.validate()
    for t in targets: t.validate()
    # Cap active subcarriers for interactive performance; report effective bandwidth.
    df = cfg.scs_khz * 1000.
    nsc = max(64, min(2048, int(cfg.channel_bandwidth_mhz*1e6/df)))
    ns = cfg.n_symbols
    fc = cfg.frequency_ghz*1e9
    lam = C/fc
    T = cfg.integration_ms/1000.
    times = np.arange(ns)*T/ns
    tones = np.arange(nsc)*df
    rng = np.random.default_rng(cfg.seed+frame*10007)
    pilots = np.exp(1j*(np.pi/2)*rng.integers(0,4,size=(ns,nsc)))
    H = np.zeros((ns,nsc), dtype=np.complex64)
    truth = []
    for target in targets:
        r = max(5.,target.range_m+target.radial_velocity_mps*.4*frame)
        tau = 2*r/C
        fd = 2*target.radial_velocity_mps/lam
        snr_db = (cfg.sensing_snr_db_at_100m+cfg.tx_power_dbm-43+
                  cfg.antenna_gain_dbi-16+target.rcs_dbsm-
                  40*np.log10(r/100))
        amp = 10**(snr_db/20)
        H += amp*(np.exp(1j*2*np.pi*times*fd)[:,None] *
                  np.exp(-1j*2*np.pi*tones*tau)[None,:]).astype(np.complex64)
        truth.append({"label":target.label,"scenario_type":target.category,
                      "range_m":round(r,2),"radial_velocity_mps":target.radial_velocity_mps,
                      "echo_snr_reference_db":round(float(snr_db),2)})
    white = (rng.normal(size=(ns,nsc))+1j*rng.normal(size=(ns,nsc)))/np.sqrt(2)
    received = pilots*(H+white.astype(np.complex64))
    equalized = received*np.conj(pilots)
    # Delay IFFT, slow-time Doppler FFT. No claim of NR PRS waveform scheduling.
    rd = np.fft.fftshift(np.fft.fft(
        np.fft.ifft(equalized*np.hanning(nsc)[None,:],axis=1)*
        np.hanning(ns)[:,None],axis=0),axes=0)
    spectrum = 20*np.log10(np.maximum(np.abs(rd[:,:nsc//2]),1e-12))
    ranges = np.arange(nsc//2)*C/(2*nsc*df)
    fd_bins = np.fft.fftshift(np.fft.fftfreq(ns,d=T/ns))
    velocities = fd_bins*lam/2
    floor = float(np.median(spectrum))
    candidates = np.argwhere(spectrum>=floor+13)
    candidates = sorted(candidates,key=lambda ij:spectrum[ij[0],ij[1]],reverse=True)
    peaks = []
    for i,j in candidates:
        if any(abs(i-p[0])<=2 and abs(j-p[1])<=3 for p in peaks): continue
        peaks.append((int(i),int(j)))
        if len(peaks)>=min(12,max(3,len(targets)*2)): break
    detections = [{"range_m":round(float(ranges[j]),2),
                   "radial_velocity_mps":round(float(velocities[i]),2),
                   "peak_above_median_db":round(float(spectrum[i,j]-floor),1)}
                  for i,j in peaks]
    metrics = {"active_subcarriers":nsc,"n_sensing_symbols":ns,
               "effective_sensing_bandwidth_mhz":round(nsc*df/1e6,3),
               "ideal_range_resolution_m":round(C/(2*nsc*df),3),
               "max_unambiguous_range_m":round(C/(2*df),2),
               "velocity_bin_spacing_mps":round(lam/(2*T),3),
               "max_unambiguous_abs_velocity_mps":round(lam*ns/(4*T),2),
               "threshold_above_median_db":13,
               "noise_floor_relative_db":round(floor,1)}
    return {"range_axis_m":ranges,"velocity_axis_mps":velocities,
            "heatmap_db":spectrum,"detections":detections,
            "truth":truth,"metrics":metrics,
            "waveform_info":{"mapping":"research QPSK pilot grid; not NR PRS/CSI-RS/SRS",
                             "fft":"delay IFFT and slow-time Doppler FFT","frame":frame}}

def communications(cfg: WaveformConfig, frame: int = 0):
    """Illustrative coexistence link model, not 3GPP transport block simulation."""
    cfg.validate()
    d = cfg.ue_distance_m
    fspl = 20*np.log10(4*np.pi*d*cfg.frequency_ghz*1e9/C)
    loss = fspl+20+12*np.log10(max(1,d/100))
    rx = cfg.tx_power_dbm+cfg.antenna_gain_dbi-loss+1.2*np.sin(frame*.25)
    noise = -174+10*np.log10(cfg.channel_bandwidth_mhz*1e6)+cfg.noise_figure_db
    interference = noise+3+8*cfg.traffic_load
    lin = 10**(rx/10)/(10**(noise/10)+10**(interference/10))
    sinr = 10*np.log10(max(1e-18,lin))
    eta = np.clip(.64*np.log2(1+lin),.02,7.4)
    throughput = cfg.channel_bandwidth_mhz*eta*.72*(1-cfg.sensing_share)*cfg.traffic_load
    return {"rx_power_dbm":round(float(rx),2),
            "sinr_db":round(float(sinr),2),
            "throughput_mbps":round(float(throughput),2),
            "spectral_efficiency_proxy_bps_hz":round(float(eta),3),
            "cqi_proxy":int(np.clip(round((sinr+6)/2.4),0,15)),
            "mcs_proxy":int(np.clip(round((sinr+9)*.75),0,28)),
            "bler_proxy_pct":round(float(100/(1+np.exp(np.clip((sinr-1)/2,-40,40)))),2),
            "latency_proxy_ms":round(5+12*cfg.traffic_load/(1.1-cfg.traffic_load)+3*cfg.sensing_share,2),
            "sensing_resource_pct":round(cfg.sensing_share*100,1)}
