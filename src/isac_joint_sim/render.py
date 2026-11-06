"""Export deterministic waveform-based ISAC research visualizations to GIF / MP4."""
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def render_frame(result,cfg):
    w,h=1280,720
    im=Image.new("RGB",(w,h),(10,30,49))
    d=ImageDraw.Draw(im)
    fpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(fpath,20) if Path(fpath).exists() else ImageFont.load_default()
    boldpath="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    bold=ImageFont.truetype(boldpath,29) if Path(boldpath).exists() else font
    d.text((40,23),"ISAC  |  SIMULTANEOUS SENSING + COMMUNICATIONS",font=bold,fill="white")
    d.text((42,75),f"{cfg.frequency_ghz:.2f} GHz  |  {cfg.channel_bandwidth_mhz:.0f} MHz  |  {cfg.scs_khz} kHz SCS  |  Frame {result['waveform_info']['frame']}",font=font,fill=(42,218,232))
    heat=result["heatmap_db"]
    lo,hi=np.percentile(heat,[10,99.8])
    v=np.clip((heat-lo)/max(hi-lo,1e-6),0,1)
    rgb=np.uint8(np.stack([20+225*v,28+190*np.sqrt(v),70+95*(1-v)],axis=-1))
    image=Image.fromarray(rgb).resize((750,465),Image.Resampling.BILINEAR)
    im.paste(image,(40,135))
    d.rectangle((40,135,790,600),outline=(32,216,228),width=2)
    d.text((42,610),"FFT Range-Doppler map (synthetic echoes)",font=font,fill="white")
    d.rectangle((819,135,1242,600),outline=(32,216,228),width=2)
    d.text((839,158),"DETECTED PEAKS",font=bold,fill="white")
    y=220
    for p in result["detections"][:6]:
        d.text((838,y),f"R={p['range_m']:.1f} m    v={p['radial_velocity_mps']:.1f} m/s",font=font,fill="white")
        y+=52
    d.text((840,545),"No RF object classification",font=font,fill=(42,218,232))
    d.text((42,673),"CONVERTURE | Research waveform experiment; not a field measurement",font=font,fill="white")
    return im

def export_gif(results,cfg,fps=4):
    images=[render_frame(r,cfg) for r in results]
    out=BytesIO()
    images[0].save(out,format="GIF",save_all=True,append_images=images[1:],
                   duration=max(100,int(1000/fps)),loop=0,optimize=True)
    return out.getvalue()

def export_mp4(results,cfg,fps=4):
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)
        for i,r in enumerate(results):render_frame(r,cfg).save(folder/f"frame_{i:04d}.png")
        out=folder/"demo.mp4"
        cmd=["ffmpeg","-y","-loglevel","error","-framerate",str(fps),"-i",
             str(folder/"frame_%04d.png"),"-c:v","libx264","-pix_fmt","yuv420p",
             "-movflags","+faststart",str(out)]
        try:subprocess.run(cmd,check=True,capture_output=True)
        except FileNotFoundError as e:raise RuntimeError("FFmpeg not installed") from e
        except subprocess.CalledProcessError as e:raise RuntimeError(e.stderr.decode(errors="replace")) from e
        return out.read_bytes()
