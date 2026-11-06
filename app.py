"""Streamlit dashboard for joint sensing/communication experiments.
Run: streamlit run app.py
"""
from __future__ import annotations
import json
import io
import zipfile
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from isac_joint_sim import Config, simulate

st.set_page_config(page_title="ISAC Joint Simulator | Converture",layout="wide",page_icon="📡")
st.markdown('''<style>
.block-container{padding-top:1rem;max-width:1600px}
section[data-testid="stSidebar"]{background:#eef3f8}
h1,h2,h3{color:#143650}
</style>''',unsafe_allow_html=True)
st.title("📡 ISAC Joint Simulator")
st.caption("Converture Market Intelligence • interactive NR-style sensing + communications research model • v0.1")
st.info("Research demonstration, **not** a 3GPP conformance implementation or field measurement. "
        "Object classes are known scenario labels; RF returns alone do not establish reliable object type identification. "
        "CQI, MCS, BLER, latency and detection visibility are illustrative proxies.")

with st.sidebar:
    st.header("Radio settings")
    presets={"Mid-band n78 (concept)":(3.5,80,30),"2.6 GHz (concept)":(2.6,60,30),
             "FR2 mmWave (concept)":(28.,200,120),"Custom":None}
    selection=st.selectbox("Preset",list(presets))
    preset=presets[selection]
    f=st.number_input("Carrier frequency (GHz)",0.7,52.6,float(preset[0] if preset else 3.5),.1)
    bw=st.slider("Channel bandwidth (MHz)",5,400,int(preset[1] if preset else 80),5)
    scs_options=[15,30,60,120,240]
    scs=st.selectbox("NR-style subcarrier spacing (kHz)",scs_options,index=scs_options.index(preset[2] if preset else 30))
    ptx=st.slider("Transmit power (dBm)",0,60,43)
    ant=st.slider("gNB antenna gain (dBi)",0,35,16)
    ue_gain=st.slider("UE antenna gain (dBi)",-5,20,0)
    ue_d=st.slider("UE distance (m)",25,5000,450,25)
    nf=st.slider("Receiver noise figure (dB)",2,15,7)
    ple=st.slider("Path loss exponent",1.8,4.5,2.7,.1)
    st.header("Joint resource use")
    sense_pct=st.slider("Sensing resource share (%)",1,60,10)
    traffic_pct=st.slider("Communication offered load (%)",0,100-sense_pct,min(55,100-sense_pct))
    integration_ms=st.slider("Sensing integration time (ms)",10,1000,100,10)
    st.header("Target scenario")
    tt=st.selectbox("Primary object (truth label)",["Vehicle","Drone","Pedestrian"])
    tr=st.slider("Initial target distance (m)",10,3000,240,10)
    tv=st.slider("Radial speed (m/s)",-80,80,-9)
    rcs=st.slider("Target radar cross-section (dBsm)",-30,30,10)
    secondary=st.checkbox("Include second object",True)
    seed=st.number_input("Random seed",0,100000,7)

cfg=Config(frequency_ghz=f,bandwidth_mhz=float(bw),scs_khz=int(scs),
           tx_power_dbm=float(ptx),tx_gain_dbi=float(ant),ue_gain_dbi=float(ue_gain),
           ue_distance_m=float(ue_d),noise_figure_db=float(nf),path_loss_exponent=float(ple),
           sensing_fraction=sense_pct/100,traffic_load=traffic_pct/100,
           sensing_integration_ms=float(integration_ms),target_distance_m=float(tr),
           target_radial_mps=float(tv),target_rcs_dbsm=float(rcs),target_type=tt,
           secondary_target=secondary,seed=int(seed))

st.markdown("### Simulation timeline")
frame=st.slider("Frame (each increment advances 0.4 seconds)",0,50,0)
result=simulate(cfg,frame)
comm=result['communications']; sense=result['sensing']; met=sense['metrics']

cols=st.columns(6)
for col,(label,value) in zip(cols,[
    ("Received power",f"{comm['received_power_dbm']:.1f} dBm"),
    ("SINR",f"{comm['sinr_db']:.1f} dB"),
    ("Throughput",f"{comm['throughput_mbps']:.1f} Mbps"),
    ("Range resolution",f"{met['range_resolution_m']:.2f} m"),
    ("Velocity resolution",f"{met['velocity_resolution_mps']:.2f} m/s"),
    ("Visible objects",str(len(sense['detections']))) ]):
    col.metric(label,value)

left,right=st.columns([1.2,1],gap="large")
with left:
    st.subheader("Range–Doppler sensing display")
    heat=sense['heatmap']
    fig=go.Figure(go.Heatmap(x=sense['range_axis_m'],y=sense['velocity_axis_mps'],z=heat,
                             colorscale="Viridis",colorbar={"title":"Relative echo (dB)"},
                             hovertemplate="Range %{x:.1f} m<br>Radial velocity %{y:.1f} m/s<br>Echo %{z:.1f} dB<extra></extra>"))
    if sense['detections']:
        fig.add_trace(go.Scatter(x=[d['range_m'] for d in sense['detections']],
                                 y=[d['radial_mps'] for d in sense['detections']],
                                 mode='markers+text',text=[d['id'] for d in sense['detections']],
                                 textposition='top center',
                                 marker={"symbol":"x","size":16,"color":"white","line":{"width":2,"color":"black"}},
                                 name="Estimated detections"))
    fig.update_layout(height=440,margin=dict(l=20,r=15,t=25,b=30),
                      xaxis_title="Round-trip estimated range (m)",yaxis_title="Radial speed (m/s)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Synthetic range–Doppler likelihood (Gaussian point-spread functions and seeded noise). "
               "No raw NR waveform or certified detector is implemented in v0.1.")
    st.subheader("Target measurements")
    if sense['detections']:
        st.dataframe(pd.DataFrame(sense['detections']).rename(columns={
            'scenario_label':'Object type (scenario truth)', 'visibility_score':'Visibility score (uncalibrated)' }),
            use_container_width=True,hide_index=True)
    else:
        st.warning("No targets cross the model's detection threshold at these parameters.")
    st.caption("'Object type' is an input scenario label, **not** an RF-derived classification claim.")

with right:
    st.subheader("Communications performance")
    kpis=pd.DataFrame([{"Metric":k.replace('_',' ').capitalize(),"Value":v} for k,v in comm.items()])
    st.dataframe(kpis,use_container_width=True,hide_index=True,height=345)
    alloc=go.Figure(go.Bar(x=[comm['resources_for_sensing_pct'],comm['resources_for_communication_pct']],
                           y=['Sensing reserved','Communications available'],orientation='h',
                           marker_color=['#16a0b3','#1c4a72'],text=[f"{comm['resources_for_sensing_pct']}%",f"{comm['resources_for_communication_pct']}%"],textposition='inside'))
    alloc.update_layout(height=195,margin=dict(l=15,r=20,t=20,b=15),xaxis=dict(range=[0,100],title='Percent of resources'),showlegend=False)
    st.plotly_chart(alloc,use_container_width=True)
    st.subheader("RF interpretation")
    st.write(f"At **{f:.2f} GHz**, wavelength is **{met['wavelength_m']*100:.2f} cm**. "
             f"With **{bw} MHz** sensing bandwidth, ideal range resolution is "
             f"**{met['range_resolution_m']:.2f} m**. A **{integration_ms:.0f} ms** integration interval "
             f"implies idealized velocity resolution of **{met['velocity_resolution_mps']:.2f} m/s**.")
    st.write(f"For the selected primary target, nominal round-trip delay is "
             f"**{met['round_trip_delay_us']:.2f} μs**, and monostatic Doppler shift is "
             f"**{met['doppler_hz']:.1f} Hz**.")

st.markdown("### Trade-off sweep: sensing resources versus communications throughput")
shares=np.linspace(.01,min(.6,1-cfg.traffic_load),25)
rows=[]
for fraction in shares:
    x=Config(**{**asdict(cfg),'sensing_fraction':float(fraction)})
    s=simulate(x,frame)
    rows.append((fraction*100,s['communications']['throughput_mbps']))
trade=go.Figure(go.Scatter(x=[r[0] for r in rows],y=[r[1] for r in rows],mode='lines+markers',line_color='#13698f'))
trade.update_layout(height=290,xaxis_title='Sensing reserved resource share (%)',yaxis_title='Illustrative user throughput (Mbps)',margin=dict(l=20,r=20,t=20,b=25))
st.plotly_chart(trade,use_container_width=True)
st.caption("This is a simplified scheduling trade-off, not a standardized 3GPP resource allocation simulator. Real performance depends on NR numerology, TDD patterns, antenna arrays, receiver processing, scheduler and channels.")

with st.expander("Model methods, constraints and standardization status"):
    st.markdown('''
- **NR connection:** adjustable carrier frequency, bandwidth, 15/30/60/120/240 kHz SCS and OFDM-inspired sensing. No 3GPP TS 38.211 physical channel waveform mapping.
- **Sensing:** ideal range resolution c/(2B), monostatic Doppler 2v/λ, integration-based velocity resolution, an R^-4 echo scaling proxy and simulated Gaussian range–Doppler likelihood.
- **Communications:** log-distance loss anchored to free-space loss at 1 m, thermal noise, illustrative load-based interference, bounded Shannon-style spectral efficiency, and simplified resource sharing.
- **Object type:** supplied by the scenario; model does **not** classify vehicle versus drone from RF signatures. That requires measured or physics-rich signatures and validated classifiers.
- **Research only:** no multipath channel model, clutter, multi-antenna angle estimation, exact PDSCH/PUSCH/PRS/SRS/CSI-RS mapping, link adaptation, 3GPP conformance or field-validated inference.
- **Standards context:** 3GPP TR 38.765 Release 20 is a study on ISAC for NR; ETSI GR ISC 003 studies 6G ISAC system/RAN architectures. These are not evidence that the prototype is standards-compliant.
''')

# Download files explicitly to support reproducibility of marketing demonstrations.
safe={"config":result['config'],"frame":frame,"communications":comm,
      "sensing_metrics":met,"detections":sense['detections']}
st.download_button("Download JSON of this frame",json.dumps(safe,indent=2),
                   file_name=f"isac_frame_{frame:03d}.json",mime="application/json")

st.caption("© Converture Inc. • ISAC Economics and Commercialization • Research demonstrator")
