import numpy as np
import pytest
from isac_joint_sim import Config, compute_communications, sensing_frame, sensing_metrics, simulate

def test_deterministic():
    a=simulate(Config(seed=27),3); b=simulate(Config(seed=27),3)
    assert a['communications']==b['communications']
    assert a['sensing']['detections']==b['sensing']['detections']
    np.testing.assert_array_equal(a['sensing']['heatmap'],b['sensing']['heatmap'])

def test_bandwidth_increases_range_resolution():
    assert sensing_metrics(Config(bandwidth_mhz=100))['range_resolution_m'] < sensing_metrics(Config(bandwidth_mhz=20))['range_resolution_m']

def test_frequency_changes_wavelength():
    assert sensing_metrics(Config(frequency_ghz=28))['wavelength_m'] < sensing_metrics(Config(frequency_ghz=3.5))['wavelength_m']

def test_load_higher_reduces_sinr():
    low=compute_communications(Config(traffic_load=.2),0)
    high=compute_communications(Config(traffic_load=.8),0)
    assert low['sinr_db']>high['sinr_db']

def test_resource_allocation_reduces_rate():
    small=compute_communications(Config(sensing_fraction=.05),1)
    big=compute_communications(Config(sensing_fraction=.4),1)
    assert small['throughput_mbps']>big['throughput_mbps']

def test_detection_changes_with_motion():
    cfg=Config(target_distance_m=240,target_radial_mps=-10)
    d0=sensing_frame(cfg,0)['objects'][0]['range_m']
    d8=sensing_frame(cfg,8)['objects'][0]['range_m']
    assert d8<d0

def test_invalid_allocation():
    with pytest.raises(ValueError): Config(traffic_load=.9,sensing_fraction=.2).validate()

def test_simulation_shapes():
    r=simulate(Config(),4)
    assert r['sensing']['heatmap'].shape==(120,180)
    assert len(r['sensing']['range_axis_m'])==180
