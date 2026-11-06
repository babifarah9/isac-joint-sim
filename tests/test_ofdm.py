import numpy as np
import pytest
from isac_joint_sim.ofdm import WaveformConfig,Target,simulate_ofdm,communications

def test_bandwidth_improves_range_resolution():
    low=simulate_ofdm(WaveformConfig(channel_bandwidth_mhz=10,n_symbols=64),[Target("T",75,0)])
    high=simulate_ofdm(WaveformConfig(channel_bandwidth_mhz=50,n_symbols=64),[Target("T",75,0)])
    assert high["metrics"]["ideal_range_resolution_m"] < low["metrics"]["ideal_range_resolution_m"]

def test_deterministic_heldout_frame():
    cfg=WaveformConfig(channel_bandwidth_mhz=10,n_symbols=64,seed=5)
    a=simulate_ofdm(cfg,[Target("A",80,2)],2)
    b=simulate_ofdm(cfg,[Target("A",80,2)],2)
    np.testing.assert_array_equal(a["heatmap_db"],b["heatmap_db"])
    assert a["detections"]==b["detections"]

def test_range_fft_peak():
    cfg=WaveformConfig(channel_bandwidth_mhz=40,scs_khz=60,n_symbols=128,
                       integration_ms=10,sensing_snr_db_at_100m=40)
    r=simulate_ofdm(cfg,[Target("A",110,0,10)])
    assert any(abs(x["range_m"]-110)<15 for x in r["detections"])

def test_doppler_fft_peak():
    cfg=WaveformConfig(frequency_ghz=3.5,channel_bandwidth_mhz=30,scs_khz=30,
                       n_symbols=128,integration_ms=20,sensing_snr_db_at_100m=40)
    r=simulate_ofdm(cfg,[Target("A",100,4,10)])
    assert any(abs(x["radial_velocity_mps"]-4)<3 for x in r["detections"])

def test_two_targets():
    cfg=WaveformConfig(channel_bandwidth_mhz=60,scs_khz=60,n_symbols=128,
                       integration_ms=20,sensing_snr_db_at_100m=50)
    r=simulate_ofdm(cfg,[Target("A",70,-3),Target("B",200,5)])
    assert any(abs(x["range_m"]-70)<10 for x in r["detections"])
    assert any(abs(x["range_m"]-200)<10 for x in r["detections"])

def test_resource_allocation_cost():
    a=communications(WaveformConfig(sensing_share=.05))
    b=communications(WaveformConfig(sensing_share=.4))
    assert b["throughput_mbps"]<a["throughput_mbps"]

def test_invalid_frequency():
    with pytest.raises(ValueError):simulate_ofdm(WaveformConfig(frequency_ghz=90),[])
