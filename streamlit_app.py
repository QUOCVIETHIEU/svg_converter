import base64
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

# ---------------- Config ----------------
st.set_page_config(page_title="PNG → SVG", layout="wide")
st.set_page_config(
    page_title="PNG → SVG", 
    page_icon="assets/icon_launcher.png", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

CONVERTER_PATH = Path("converter.py")  # change if your script name differs

# ---------------- Styles ----------------
CSS = """
<style>
:root {
  --panel-radius: 18px;
  --border-color: #e5e7eb;
  --text-light: #6b7280;
  --brand: #0ea5e9;
  --btn-blue: #22a6ff;
  --btn-blue-hover: #0ea5e9;
  --btn-muted: #e5e7eb;
  --h1: 42px;
}
body { padding-bottom: 80px; }
.block-container {padding-top: 24px; padding-bottom: 24px; max-width: 1300px;}
.hero h1 {font-size: var(--h1); font-weight: 700; margin: 0 0 8px 0;}
.left-panel {border: 1px solid var(--border-color); border-radius: var(--panel-radius); padding: 8px; background: #fff; height: 400px; display: flex; align-items: center; justify-content: center; flex-direction: column; overflow: auto;}
.dropzone {border: 2px dashed #d1d5db; border-radius: 16px; min-height: 360px; display: flex; align-items: center; justify-content: center; text-align: center; color: #111827; background: #fcfcfd;}
.dropzone h3 {font-size: 24px; margin: 8px 0;}
.dropzone p { color: #6b7280; margin: 8px 0 18px; }
.dropzone .btn {background: #1677ff; color: #fff; border: none; border-radius: 999px; padding: 12px 22px; font-weight: 600; cursor: pointer;}
.dropzone .btn:hover { background: #0e5ed1; }
[data-testid="stFileUploader"] > section { border: 0 !important; padding: 20px !important; }
[data-testid="stFileUploader"] div[class*="uploadFileName"] { display: none !important; }
[data-testid="stFileUploader"] label { color: #1677ff; cursor: pointer; text-decoration: underline; }
[data-testid="stFileUploaderDropzone"] { padding: 20px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { padding: 15px !important; }
[data-testid="stFileUploader"] .st-emotion-cache-fis6aj { display: flex !important; justify-content: center !important; }
[data-testid="stFileUploader"] ul { display: flex !important; justify-content: center !important; }
.right-panel h3 { font-size: 24px; margin-bottom: 6px; }
.section-caption { font-weight: 700; color: #6b7280; margin-top: 16px; margin-bottom: 6px; }
a.reset-link { font-size: 13px; color: #6b7280; text-decoration: underline; float: right; margin-top: 4px; }
.stNumberInput > div > div, .stTextInput > div > div { border-radius: 4px !important; }
.actions { display: flex; gap: 12px; }
button[data-testid="baseButton-primary"] { background: #19c1ff !important; color: #fff !important; border-radius: 999px !important; height: 42px; font-weight: 600; }
button.download-btn { background: #e5e7eb !important; color: #111827 !important; border-radius: 999px !important; height: 42px; font-weight: 600; }

/* Zoom buttons styling */
[data-testid="column"] button[data-testid="baseButton-secondary"] { 
    background: #f8fafc !important; 
    color: #374151 !important; 
    border: 1px solid #e5e7eb !important; 
    border-radius: 8px !important; 
    height: 36px !important; 
    font-size: 18px !important; 
    font-weight: 600 !important; 
    transition: all 0.2s ease !important;
}
[data-testid="column"] button[data-testid="baseButton-secondary"]:hover { 
    background: #e2e8f0 !important; 
    border-color: #cbd5e1 !important; 
}
.preview-wrap { border: 1px dashed #e5e7eb; border-radius: 16px; height: 256px; display:flex; align-items:center; justify-content:center; background:#fff; }
.toolbar { display:flex; align-items:center; gap:10px; justify-content:center; margin-top: 10px; color:#111827;}
.toolbar button { background:#f1f5f9; border:1px solid #e5e7eb; border-radius:8px; padding:6px 10px;}
.zoom-label { min-width: 60px; text-align: center; }
.zoom-controls { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 16px; }
.zoom-btn { min-width: 36px; height: 36px; border-radius: 8px; border: 1px solid #e5e7eb; background: #f8fafc; color: #374151; font-size: 18px; font-weight: 600; display: flex; align-items: center; justify-content: center; cursor: pointer; }
.zoom-btn:hover { background: #e2e8f0; border-color: #cbd5e1; }
.zoom-display { min-width: 60px; padding: 8px 12px; text-align: center; font-weight: 500; color: #374151; background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; }
/* Styling cho 2 nút Upload New và Download File */
.upload-new-btn button[data-testid="baseButton-secondary"] { 
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important; 
    color: #fff !important; 
    border: none !important; 
    border-radius: 999px !important; 
    height: 42px !important; 
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3) !important;
}
.upload-new-btn button[data-testid="baseButton-secondary"]:hover { 
    background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important; 
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(245, 158, 11, 0.4) !important;
}
.download-file-btn button[data-testid="baseButton-secondary"] { 
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important; 
    color: #fff !important; 
    border: none !important; 
    border-radius: 999px !important; 
    height: 42px !important; 
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3) !important;
}
.download-file-btn button[data-testid="baseButton-secondary"]:hover { 
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important; 
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(16, 185, 129, 0.4) !important;
}

/* Hide Streamlit default header and anchor links in headings */
[data-testid="stHeaderActionElements"],
[data-testid="stHeadingWithActionElements"] a,
h1 a[href^="#"],
h2 a[href^="#"],
h3 a[href^="#"],
h4 a[href^="#"],
h5 a[href^="#"],
h6 a[href^="#"] {
  display: none !important;
  pointer-events: none !important;
}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Helpers ----------------
def _svg_component(svg_text: str, scale: float=1.0, height=256):
    svg_b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    html = f"""
    <div class="preview-wrap">
      <div id="canvas" style="transform:scale({scale});transform-origin:center center;">
        <object type="image/svg+xml" data="data:image/svg+xml;base64,{svg_b64}" style="max-width:100%;max-height:{height}px"></object>
      </div>
    </div>
    """
    st.components.v1.html(html, height=height+6, scrolling=False)

def run_cli(args):
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def convert_with_params(in_bytes: bytes, params: dict) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / "input.png"
        out_path = td / "output.svg"
        in_path.write_bytes(in_bytes)
        cmd = [
            "python", str(CONVERTER_PATH),
            "--input", str(in_path),
            "--output", str(out_path),
            "--mode", params["mode"],
            "--k", str(params["k"]),
            "--min-area", str(params["min_area"]),
            "--area-max-ratio", str(params["area_max_ratio"]),
            "--super-sample", str(params["super_sample"]),
            "--decimate", str(params["decimate"]),
            "--subpixel-sigma", str(params["subpixel_sigma"]),
            "--repair-clean", str(params["repair_clean"]),
            "--corner-eps", str(params["corner_eps"]),
            "--line-tol", str(params["line_tol"]),
            "--max-err", str(params["max_err"]),
            "--morph-close", str(params["morph_close"]),
            "--morph-open", str(params["morph_open"]),
            "--curve-sigma", str(params["curve_sigma"]),
        ]
        if not params["auto_crop"]:
            cmd.append("--no-auto-crop")
        if not params["repair"]:
            cmd.append("--no-repair")
        if params["crop"]:
            cmd += ["--crop", params["crop"]]

        rc, out, err = run_cli(cmd)
        if rc != 0 or not out_path.exists():
            raise RuntimeError(err or out or "Conversion failed.")
        return out_path.read_text(encoding="utf-8"), out

# ---------------- Defaults ----------------
DEFAULTS = dict(
    mode="auto", k=4,
    min_area=80, area_max_ratio=0.92,
    super_sample=4, decimate=1, subpixel_sigma=0.8,
    auto_crop=True, crop="",
    morph_close=4, morph_open=1,
    repair=True, repair_clean=0.01,
    corner_eps=0.9, line_tol=0.10, max_err=0.26,
    curve_sigma=1.4,
)

if "params" not in st.session_state:
    st.session_state.params = DEFAULTS.copy()
if "svg" not in st.session_state:
    st.session_state.svg = None
if "zoom" not in st.session_state:
    st.session_state.zoom = 1.0

# ---------------- Header ----------------
# Đọc và hiển thị icon với tiêu đề
try:
    import base64
    with open("assets/app_launcher.svg", "rb") as f:
        icon_data = f.read()
    icon_b64 = base64.b64encode(icon_data).decode()
    
    st.markdown(f'''
    <div class="hero">
        <h1 style="display: flex; flex-direction: column; align-items: center; gap: 12px;">
            <img src="data:image/svg+xml;base64,{icon_b64}" 
                 style="width: 96px; height: auto;" 
                 alt="App Icon">
            Convert your image to an SVG
        </h1>
    </div>
    ''', unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback nếu không tìm thấy icon
    st.markdown('<div class="hero"><h1>Convert your image to an SVG</h1></div>', unsafe_allow_html=True)

# ---------------- Layout ----------------
left, right = st.columns([1.25, 1])

# LEFT: upload + preview
with left:
    # Hiển thị file uploader hoặc 2 nút tùy theo trạng thái
    if st.session_state.svg:
        # Sau khi convert - hiển thị 2 nút Upload New và Download (nhỏ gọn)
        empty1, action_col1, gap, action_col2, empty2 = st.columns([1, 1.5, 0.2, 1.5, 1])
        with action_col1:
            # Nút Upload New
            upload_clicked = st.button("Upload New", 
                                     use_container_width=True,
                                     help="Upload a new image")
            if upload_clicked:
                st.session_state.svg = None
                st.session_state.zoom = 1.0
                st.rerun()
            
        with action_col2:
            # Nút Download giống như nút Process - dùng tên file gốc
            uploaded_file = st.session_state.get('up')
            if uploaded_file:
                # Lấy tên file gốc và thay extension thành .svg
                original_name = uploaded_file.name
                svg_filename = original_name.rsplit('.', 1)[0] + '.svg'
            else:
                svg_filename = "converted_image.svg"
                
            st.download_button("Download File", 
                             data=st.session_state.svg.encode("utf-8"),
                             file_name=svg_filename, 
                             mime="image/svg+xml",
                             use_container_width=True,
                             type="primary",
                             help="Download the converted SVG file")
    else:
        # Trước khi convert - hiển thị file uploader
        uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"], key="up", label_visibility="hidden")
    # Khung preview hiển thị sau khi upload
    uploaded_file = st.session_state.get('up')
    
    if st.session_state.svg:
        # Hiển thị kết quả SVG trong left-panel (không có border dash)
        html_svg = f"""
        <div class="left-panel">
            <div style="
                border-radius: 16px; 
                min-height: {max(256, int(256 * st.session_state.zoom))}px;
                min-width: {max(400, int(400 * st.session_state.zoom))}px;
                display: flex; 
                align-items: center; 
                justify-content: center; 
                background: #fff;
                overflow: visible;
                position: relative;
            ">
                <div id="canvas" style="
                    transform: scale({st.session_state.zoom}); 
                    transform-origin: center center;
                    width: 240px;
                    height: 240px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <object type="image/svg+xml" 
                            data="data:image/svg+xml;base64,{base64.b64encode(st.session_state.svg.encode("utf-8")).decode("ascii")}" 
                            style="width: 100%; height: 100%; object-fit: contain;"></object>
                </div>
            </div>
        </div>
        """
        st.markdown(html_svg, unsafe_allow_html=True)
        
        # Custom styled success message với thông tin file input gốc
        uploaded_file = st.session_state.get('up')
        if uploaded_file:
            file_name = uploaded_file.name
            # Lấy kích thước file input gốc
            file_size_kb = uploaded_file.size / 1024
            if file_size_kb < 1024:
                size_text = f" ({file_size_kb:.1f}KB)"
            else:
                size_text = f" ({file_size_kb/1024:.1f}MB)"
        else:
            file_name = "image"
            size_text = "unknown"
            
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%); 
            border: 0.5px solid #10b981; 
            border-radius: 12px; 
            padding: 16px 20px; 
            margin: 16px 0; 
            display: flex; 
            align-items: center; 
            gap: 12px;
            box-shadow: 0 2px 4px rgba(16, 185, 129, 0.1);
        ">
            <div style="
                font-size: 24px; 
                line-height: 1;
            ">✅</div>
            <div style="
                color: #047857; 
                font-weight: 600; 
                font-size: 15px;
            ">Conversion completed file '{file_name}' {size_text}.</div>
        </div>
        """, unsafe_allow_html=True)
    elif uploaded_file:
        # Hiển thị ảnh đã upload trong left-panel
        img_b64 = base64.b64encode(uploaded_file.read()).decode("ascii")
        uploaded_file.seek(0)  # Reset file pointer
        html = f"""
        <div class="left-panel">
            <div style="border-radius: 16px; height: 256px; display:flex; align-items:center; justify-content:center; background:#fff;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%;max-height:256px;object-fit:contain;border-radius:12px;">
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        # Hiển thị khung preview mặc định khi chưa upload
        html_default = """
        <div class="left-panel">
            <div style="border-radius: 16px; height: 256px; display:flex; align-items:center; justify-content:center; background:#fff;">
                <div style="text-align:center; color:#6b7280;">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="currentColor" style="margin-bottom: 16px; opacity: 0.3;">
                        <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
                    </svg>
                    <h3 style="margin: 0 0 8px 0; font-size: 18px;">Preview Area</h3>
                    <p style="margin: 0; font-size: 14px;">Upload an image to see preview</p>
                </div>
            </div>
        </div>
        """
        st.markdown(html_default, unsafe_allow_html=True)

    # Zoom controls (chỉ hiển thị khi có SVG) - chỉ chiếm 1/2 chiều rộng
    if st.session_state.svg:
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        
        # Tạo bố cục để zoom controls chỉ chiếm 1/2 chiều rộng cột trái
        empty1, zoom_container, empty2 = st.columns([1, 2, 1])
        
        with zoom_container:
            # Tạo 3 cột nhỏ trong container zoom
            zcol1, zcol2, zcol3 = st.columns([1, 1, 1])
            
            with zcol1:
                if st.button("−", key="zoom_minus", help="Zoom out", use_container_width=True):
                    st.session_state.zoom = max(0.2, st.session_state.zoom - 0.1)
                    st.rerun()
            
            with zcol2:
                # Button để reset zoom về 100% khi click
                zoom_reset = st.button(f"{int(st.session_state.zoom*100)}%", 
                                     key="zoom_reset", 
                                     help="Click to reset zoom to 100%",
                                     use_container_width=True)
                if zoom_reset:
                    st.session_state.zoom = 1.0
                    st.rerun()
            
            with zcol3:
                if st.button("＋", key="zoom_plus", help="Zoom in", use_container_width=True):
                    st.session_state.zoom = min(10.0, st.session_state.zoom + 0.1)
                    st.rerun()

    # Progress bar placeholder trong left column
    progress_placeholder = st.empty()
    
    # Processing logic trong left column
    uploaded_file = st.session_state.get('up')
    if st.session_state.get('process_started', False):
        if not uploaded_file:
            progress_placeholder.warning("Please upload an image first.")
        elif not CONVERTER_PATH.exists():
            progress_placeholder.error(f"Converter script not found at: {CONVERTER_PATH}")
        else:
            try:
                # Show progress bar in left column
                progress_html = """
                <div style="
                    background: #f8fafc; 
                    border: 1px solid #e2e8f0; 
                    border-radius: 12px; 
                    padding: 8px 8px; 
                    margin: 8px 0;
                    text-align: center;
                ">
                    <div style="color: #374151; font-weight: 600; margin-bottom: 12px; font-size: 16px;">
                        ⚛ Processing your image...
                    </div>
                    <div style="
                        width: 100%; 
                        height: 4px; 
                        background: #e2e8f0; 
                        border-radius: 4px; 
                        overflow: hidden;
                        position: relative;
                    ">
                        <div style="
                            height: 100%; 
                            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%); 
                            border-radius: 4px;
                            animation: progress-animation 2s ease-in-out infinite;
                            width: 100%;
                        "></div>
                    </div>
                </div>
                <style>
                @keyframes progress-animation {
                    0% { width: 0%; }
                    50% { width: 70%; }
                    100% { width: 100%; }
                }
                </style>
                """
                progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
                
                # Process the conversion
                svg_text, log = convert_with_params(uploaded_file.read(), st.session_state.params)
                
                # Clear progress bar from left column
                progress_placeholder.empty()
                
                # Update session state
                st.session_state.svg = svg_text
                st.session_state.zoom = 1.0
                st.session_state.process_started = False
                
                # Tự động download SVG sau khi convert xong
                svg_b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
                auto_download_script = f"""
                <script>
                    const link = document.createElement('a');
                    link.href = 'data:image/svg+xml;base64,{svg_b64}';
                    link.download = 'converted_image.svg';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                </script>
                """
                st.markdown(auto_download_script, unsafe_allow_html=True)
                
                st.rerun()
            except Exception as e:
                progress_placeholder.error("Conversion failed.")
                with st.expander("Details"):
                    st.code(str(e))
                st.session_state.process_started = False

# RIGHT: params
with right:
    st.markdown("""
    <div class="right-panel" style="padding-top: 20px;">
      <h3>Processing Parameters:</h3>
    </div>
    """, unsafe_allow_html=True)

    p = st.session_state.params

    # Chia parameters thành 2 cột
    param_col1, param_col2 = st.columns(2)
    
    # Disable parameters khi đã convert xong
    is_disabled = st.session_state.svg is not None
    
    with param_col1:
        p["k"] = st.number_input("K:", min_value=1, max_value=64, value=int(p["k"]), step=1, disabled=is_disabled)
        p["min_area"] = st.number_input("Min area:", min_value=0, value=int(p["min_area"]), step=10, disabled=is_disabled)
        p["super_sample"] = st.number_input("Super sample:", min_value=1, max_value=8, value=int(p["super_sample"]), step=1, disabled=is_disabled)
        p["morph_close"] = st.number_input("Morph close:", min_value=0, max_value=25, value=int(p["morph_close"]), step=1, disabled=is_disabled)
        p["morph_open"]  = st.number_input("Morph open:",  min_value=0, max_value=25, value=int(p["morph_open"]), step=1, disabled=is_disabled)
    
    with param_col2:
        p["corner_eps"] = st.number_input("Corner eps:", min_value=0.0, max_value=5.0, value=float(p["corner_eps"]), step=0.05, disabled=is_disabled)
        p["line_tol"]   = st.number_input("Line tol:",   min_value=0.0, max_value=1.0, value=float(p["line_tol"]), step=0.01, disabled=is_disabled)
        p["max_err"]    = st.number_input("Max err:",    min_value=0.0, max_value=5.0, value=float(p["max_err"]), step=0.02, disabled=is_disabled)
        p["curve_sigma"]= st.number_input("Curve sigma:", min_value=0.0, max_value=5.0, value=float(p["curve_sigma"]), step=0.1, disabled=is_disabled)
        p["subpixel_sigma"] = st.number_input("Subpixel sigma:", min_value=0.0, max_value=5.0, value=float(p["subpixel_sigma"]), step=0.1, disabled=is_disabled)

    st.write("")
    
    # Nút Process - disable khi đã convert xong hoặc chưa upload file
    uploaded_file = st.session_state.get('up')
    is_process_disabled = is_disabled or uploaded_file is None
    
    process = st.button("Process", type="primary", use_container_width=True, disabled=is_process_disabled)
    if process:
        st.session_state.process_started = True
        st.rerun()


st.markdown("""
<div style="
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #f9f9f9;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    padding: 10px ;
    color: #666;
    font-size: 0.85em;
    z-index: 1000;
    box-shadow: none;
    backdrop-filter: none;
">
    <p style="margin: 0;">Copyright © hieuvoquoc@gmail.com (V1.02 - SVG Converter)</p>
</div>
""", unsafe_allow_html=True)
