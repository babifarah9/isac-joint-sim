"""Deterministic joint sensing + communication teaching/research simulator.

NOT a 3GPP conformance stack, commercial base station, or validated object classifier.
All link, target echo and detection-confidence models are explicitly illustrative.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

C = 299_792_458.0
K_BOLTZMANN_DBM_HZ = -174.0  # approximate thermal noise density at 290 K

@dataclass(frozen=True)
class Config:
    frequency_ghz: float = 3.5
    bandwidth_mhz: float = 80.0
    scs_khz: int = 30
    tx_power_dbm: float = 43.0
    tx_gain_dbi: float = 16.0
    ue_gain_dbi: float = 0.0
    noise_figure_db: float = 7.0
    path_loss_exponent: float = 2.7
    ue_distance_m: float = 450.0
    traffic_load: float = 0.55
    sensing_fraction: float = 0.10
    sensing_integration_ms: float = 100.0
    target_rcs_dbsm: float = 10.0
    target_distance_m: float = 240.0
    target_radial_mps: float = -9.0
    target_type: str = "Vehicle"
    secondary_target: bool = True
    seed: int = 7

    def validate(self) -> None:
        if not 0.7 <= self.frequency_ghz <= 52.6:
            raise ValueError("Frequency must be 0.7–52.6 GHz for this educational model")
        if not 5 <= self.bandwidth_mhz <= 400:
            raise ValueError("Bandwidth must be 5–400 MHz")
        if self.scs_khz not in (15,30,60,120,240):
            raise ValueError("SCS must be a standard NR-style numerology value")
        if not -30 <= self.tx_power_dbm <= 65:
            raise ValueError("Transmit power out of modeled range")
        if not -10 <= self.tx_gain_dbi <= 40 or not -10 <= self.ue_gain_dbi <= 30:
            raise ValueError("Antenna gain out of range")
        if not 0 <= self.traffic_load <= 1 or not 0.01 <= self.sensing_fraction <= 0.6:
            raise ValueError("Traffic and sensing fractions out of range")
        if self.traffic_load + self.sensing_fraction > 1:
            raise ValueError("Traffic and sensing fractions must total at most 1")
        if not 25 <= self.ue_distance_m <= 5000 or not 10 <= self.target_distance_m <= 3000:
            raise ValueError("Range out of modeled bounds")
        if not -80 <= self.target_radial_mps <= 80:
            raise ValueError("Target radial speed out of range")
        if self.target_type not in ("Vehicle","Drone","Pedestrian"):
            raise ValueError("Target type must be Vehicle, Drone, or Pedestrian")
        if not 10 <= self.sensing_integration_ms <= 1000:
            raise ValueError("Coherent integration time must be 10–1000 ms")
        if not 1.8 <= self.path_loss_exponent <= 4.5:
            raise ValueError("Path-loss exponent out of range")


def db_to_lin(db): return 10.0 ** (db / 10.0)
def lin_to_db(x): return 10.0 * math.log10(max(x,1e-20))

def compute_communications(cfg: Config, frame: int = 0) -> dict:
    """Illustrative NR-like downlink budget; RSRP not an actual 3GPP SS-RSRP measurement."""
    cfg.validate()
    f_mhz = cfg.frequency_ghz * 1000
    d_km = cfg.ue_distance_m/1000
    fspl_1m = 32.44 + 20*math.log10(f_mhz) + 20*math.log10(.001)
    loss = fspl_1m + 10*cfg.path_loss_exponent * math.log10(cfg.ue_distance_m)
    # 1.5 dB slow deterministic fade; ensures changing RF parameters are visible.
    fade = 1.5*math.sin(frame*.52 + cfg.seed*.19)
    rx_dbm = cfg.tx_power_dbm + cfg.tx_gain_dbi + cfg.ue_gain_dbi - loss + fade
    bw_hz = cfg.bandwidth_mhz*1e6
    noise_dbm = K_BOLTZMANN_DBM_HZ + 10*math.log10(bw_hz) + cfg.noise_figure_db
    # interference approximation increases with competing traffic load
    interference_dbm = noise_dbm + 4 + 10*cfg.traffic_load
    sinr_lin = db_to_lin(rx_dbm)/(db_to_lin(noise_dbm)+db_to_lin(interference_dbm))
    sinr_db = lin_to_db(sinr_lin)
    # Signal-power proxy labeled 'received power', not 3GPP RSRP
    usable_fraction = max(0.0, 1 - cfg.sensing_fraction)
    spectral_efficiency = min(7.4, max(0.05, .62*math.log2(1+sinr_lin)))
    # 0.75 generic implementation overhead; demand-limited offered traffic fraction
    throughput_mbps = cfg.bandwidth_mhz * spectral_efficiency * .75 * usable_fraction * cfg.traffic_load
    # CQI/MCS are synthetic indicators, NOT exact 3GPP table lookups
    cqi = int(np.clip(round((sinr_db+7)/2.2),0,15))
    mcs = int(np.clip(round(cqi*1.8),0,28))
    bler_pct = float(100/(1+math.exp(np.clip((sinr_db-1.5)/2.5,-50,50))))
    latency_ms = round(5+12*cfg.traffic_load/(max(0.1,1-cfg.traffic_load))+3*cfg.sensing_fraction,2)
    return {
        "received_power_dbm":round(rx_dbm,2),"path_loss_db":round(loss,2),
        "noise_dbm":round(noise_dbm,2),"sinr_db":round(sinr_db,2),
        "throughput_mbps":round(throughput_mbps,2),"cqi_proxy":cqi,
        "mcs_proxy":mcs,"bler_proxy_pct":round(bler_pct,2),
        "latency_proxy_ms":latency_ms,"spectral_efficiency_bps_hz":round(spectral_efficiency,3),
        "resources_for_sensing_pct":round(cfg.sensing_fraction*100,1),
        "resources_for_communication_pct":round((1-cfg.sensing_fraction)*100,1)
    }


def sensing_metrics(cfg:Config)->dict:
    cfg.validate()
    wavelength = C/(cfg.frequency_ghz*1e9)
    return {"wavelength_m":wavelength,
            "range_resolution_m": C/(2*cfg.bandwidth_mhz*1e6),
            "velocity_resolution_mps":wavelength/(2*cfg.sensing_integration_ms/1000),
            "max_unambiguous_range_m_idealized": C/(2*cfg.scs_khz*1000),
            "round_trip_delay_us":2*cfg.target_distance_m/C*1e6,
            "doppler_hz":2*cfg.target_radial_mps/wavelength}


def get_objects(cfg:Config, frame:int)->list[dict]:
    """Truth-labelled synthetic targets. Class labels are scenario inputs, not ML predictions."""
    cfg.validate()
    dt=.4*frame
    primary_range=max(5.0,cfg.target_distance_m+cfg.target_radial_mps*dt)
    objects=[{"id":"T01","type":cfg.target_type,"range_m":primary_range,
              "radial_mps":cfg.target_radial_mps,"rcs_dbsm":cfg.target_rcs_dbsm}]
    if cfg.secondary_target:
        secondary_type="Drone" if cfg.target_type != "Drone" else "Vehicle"
        objects.append({"id":"T02","type":secondary_type,
                        "range_m":max(5.,cfg.target_distance_m*.59+7*dt),
                        "radial_mps":7.,"rcs_dbsm":-4 if secondary_type=="Drone" else 8})
    return objects


def sensing_frame(cfg:Config, frame:int=0, num_range_bins:int=180, num_vel_bins:int=120)->dict:
    """Simplified range–Doppler likelihood map; not an OFDM FFT waveform estimator.
    The map displays Gaussian PSFs, noise and nominal radar-equation scaling.
    """
    cfg.validate()
    rng=np.random.default_rng(cfg.seed+10_007*frame)
    metrics=sensing_metrics(cfg)
    objects=get_objects(cfg,frame)
    ranges=np.linspace(0,max(400,cfg.target_distance_m*1.8),num_range_bins)
    velocities=np.linspace(-80,80,num_vel_bins)
    rr,vv=np.meshgrid(ranges,velocities)
    heat=rng.exponential(scale=.018,size=rr.shape)
    detections=[]
    range_sigma=max(metrics['range_resolution_m'],3)
    velocity_sigma=max(metrics['velocity_resolution_mps'],1.8)
    # Radar R^-4 law relative to reference, RCS dependence; not calibrated receiver SNR
    for obj in objects:
        echo_db=18 + (obj['rcs_dbsm']-10) + 40*math.log10(200/max(25,obj['range_m']))
        # Increased transmit power and integration time improves illustrative target SNR
        echo_db += .6*(cfg.tx_power_dbm-43)+10*math.log10(cfg.sensing_integration_ms/100)
        echo_db=float(np.clip(echo_db,-25,35))
        amp=min(3.,max(.03,db_to_lin(echo_db/2)/15))
        heat += amp*np.exp(-.5*((rr-obj['range_m'])/range_sigma)**2-.5*((vv-obj['radial_mps'])/velocity_sigma)**2)
        detectable=echo_db >= 0
        if detectable:
            dist_sigma=max(range_sigma, .7)
            vel_sigma=max(velocity_sigma,.4)
            measured_r=max(0.,obj['range_m']+rng.normal(0,dist_sigma*.25))
            measured_v=obj['radial_mps']+rng.normal(0,vel_sigma*.25)
            # 'confidence' is a visualization score, not calibrated classification probability
            confidence_proxy=float(np.clip((echo_db+10)/40*100,10,99))
            detections.append({"id":obj['id'],"scenario_label":obj['type'],
                               "range_m":round(measured_r,1),"radial_mps":round(measured_v,1),
                               "echo_snr_proxy_db":round(echo_db,1),
                               "visibility_score":round(confidence_proxy,1)})
    return {"range_axis_m":ranges,"velocity_axis_mps":velocities,
            "heatmap":10*np.log10(np.maximum(heat,1e-5)),"detections":detections,
            "metrics":metrics,"objects":objects}


def simulate(cfg: Config,frame:int=0)->dict:
    return {"config":asdict(cfg),"frame":frame,"communications":compute_communications(cfg,frame),
            "sensing":sensing_frame(cfg,frame)}
