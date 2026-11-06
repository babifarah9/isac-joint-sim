"""Streamlit v0.2: FFT ISAC sensing, simultaneous communications dashboard."""
from dataclasses import asdict
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from isac_joint_sim.ofdm import WaveformConfig,Target,simulate_ofdm,communications

st.set_page_config(page_title="ISAC Joint Simulator v0.2",page_icon="📡",layout="wide")
st.title("📡 ISAC Joint Simulator v0.2 — Converture")
st.caption("CP-OFDM research waveform • delayed target echoes • range/Doppler FFT • simultaneous communications")
st.warning("3GPP-aligned research only, NOT a compliant NR implementation or field validation. Object types are scenario inputs, NOT RF-inferred classifications. Sensing SNR and link KPIs are modeled assumptions.")

with st.sidebar:
    st.header("Radio")
    frequency_ghz=st.number_input("Frequency (GHz)",.7,52.6,3.5,.1)
    channel_bandwidth_mhz=st.slider("Channel bandwidth (MHz)",5,400,80,5)
    scs_khz=st.selectbox("NR-style SCS (kHz)",[15,30,60,120,240],index=1)
    tx_power_dbm=st.slider("Transmit power (dBm)",0,60,43)
    antenna_gain_dbi=st.slider("Antenna gain (dBi)",0,35,16)
    n_symbols=st.select_slider("Sensing symbols",options=[64,128,256],value=128)
    integration_ms=st.slider("Coherent observation (ms)",5,200,20,5)
    sensing_snr_db_at_100m=st.slider("Assumed sensing SNR at 100m (dB)",-20,60,26)
    st.header("Network")
    ue_distance_m=st.slider("UE range (m)",25,5000,450,25)
    noise_figure_db=st.slider("Receiver NF (dB)",2,15,7)
    sensing_share=st.slider("Sensing allocation (%)",0,60,12)/100
    traffic_load=st.slider("Communication offered load (%)",0,100,65)/100
    st.header("Sensing targets")
    scenario=st.selectbox("Object A truth label",["Drone","Vehicle","Pedestrian"])
    d1=st.slider("Object A range (m)",10,1000,110,5)
    v1=st.slider("Object A radial velocity (m/s)",-40,40,4)
    rcs=st.slider("Object A RCS (dBsm)",-25,25,10)
    two=st.checkbox("Second object",True)
    if two:
        d2=st.slider("Object B range (m)",10,1000,220,5)
        v2=st.slider("Object B radial velocity (m/s)",-40,40,-6)
    seed=st.number_input("Random seed",0,100000,7)

cfg=WaveformConfig(frequency_ghz=frequency_ghz,channel_bandwidth_mhz=channel_bandwidth_mhz,scs_khz=scs_khz,tx_power_dbm=tx_power_dbm,antenna_gain_dbi=antenna_gain_dbi,noise_figure_db=noise_figure_db,sensing_snr_db_at_100m=sensing_snr_db_at_100m,integration_ms=integration_ms,sensing_share=sensing_share,traffic_load=traffic_load,ue_distance_m=ue_distance_m,n_symbols=n_symbols,seed=int(seed))
targets=[Target("A",d1,v1,rcs,scenario)]
if two: targets.append(Target("B",d2,v2,0,"Vehicle"))
frame=st.slider("Simulation frame (0.4s increments)",0,40,0)
r=simulate_ofdm(cfg,targets,frame)
com=communications(cfg,frame)
met=r["metrics"]
a=st.columns(5)
for col,label,value in zip(a,["Range resolution","Velocity bin","SINR proxy","Throughput proxy","Detections"],[f"{met['ideal_range_resolution_m']} m",f"{met['velocity_bin_spacing_mps']} m/s",f"{com['sinr_db']} dB",f"{com['throughput_mbps']} Mbps",str(len(r['detections']))]):col.metric(label,value)
left,right=st.columns([1.3,1])
with left:
    st.subheader("FFT-derived Range–Doppler")
    fig=go.Figure(go.Heatmap(x=r["range_axis_m"],y=r["velocity_axis_mps"],z=r["heatmap_db"],colorscale="Viridis"))
    fig.add_trace(go.Scatter(x=[p["range_m"] for p in r["detections"]],y=[p["radial_velocity_mps"] for p in r["detections"]],mode="markers",marker=dict(symbol="x",size=11,color="white"),name="Peak candidates"))
    fig.update_layout(height=480,xaxis_title="Range (m)",yaxis_title="Radial velocity (m/s)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Simulated QPSK pilot grid → echo model → delay IFFT + Doppler FFT → heuristic peak pick. Not calibrated CFAR.")
    st.subheader("Estimated peaks, no class prediction")
    st.dataframe(pd.DataFrame(r["detections"]),use_container_width=True,hide_index=True)
    st.subheader("Scenario truth (not inference)")
    st.dataframe(pd.DataFrame(r["truth"]),use_container_width=True,hide_index=True)
with right:
    st.subheader("Communications")
    st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in com.items()]),use_container_width=True,hide_index=True)
    st.subheader("Waveform metrics")
    st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in met.items()]),use_container_width=True,hide_index=True)
    if any(abs(t.radial_velocity_mps)>met["max_unambiguous_abs_velocity_mps"] for t in targets):st.error("Doppler aliasing risk: velocity beyond unambiguous limit.")
    if any(t.range_m>met["max_unambiguous_range_m"] for t in targets):st.error("Range ambiguity risk.")

payload={"config":asdict(cfg),"frame":frame,"peaks":r["detections"],"truth":r["truth"],"metrics":met,"communications":com}
st.download_button("Download JSON",json.dumps(payload,indent=2),"isac_v02_frame.json","application/json")
st.caption("Research only. 3GPP TR 38.765 Release 20 is an ISAC study, not proof this model meets NR standards.")
