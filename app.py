# file: app.py
import streamlit as st
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image
from io import BytesIO

# Giả định file converter.py nằm cùng thư mục
# Import các hàm xử lý chính từ file của bạn
from converter import convert_png_to_svg, auto_crop_foreground_lab, is_mono_icon

# --- Cấu hình trang và Session State ---
st.set_page_config(layout="wide", page_title="Image to SVG Converter")

# Khởi tạo session state để lưu trữ trạng thái
if 'svg_result' not in st.session_state:
    st.session_state.svg_result = None
if 'uploaded_file_key' not in st.session_state:
    st.session_state.uploaded_file_key = str(int(time.time()))
if 'params' not in st.session_state:
    st.session_state.params = {}

# --- Giao diện người dùng ---

st.title("Convert your image to an SVG.")

# Định nghĩa các tham số mặc định dựa trên UI và file python
DEFAULT_PARAMS = {
    'mode': 'Auto',
    'k': 4,
    'min_area': 80,
    'area_max_ratio': 0.92,
    'super_sample': 4,
    'decimate': 1,
    'subpixel_sigma': 0.8,
    'morph_close': 4,
    'morph_open': 1,
    'corner_eps': 0.9,
    'line_tol': 0.1,
    'max_err': 0.26,
    'repair_clean': 0.01,
    'curve_sigma': 1.4,
}

# Hàm reset tham số
def reset_params():
    st.session_state.params = DEFAULT_PARAMS.copy()
    # Thêm một key ngẫu nhiên để force re-render các widget
    st.session_state.widget_key = f"reset_{int(time.time())}"

# Khởi tạo lần đầu
if not st.session_state.params:
    reset_params()


# Chia layout thành 2 cột
col1, col2 = st.columns([0.55, 0.45])


# --- Cột trái: Upload và hiển thị kết quả ---
with col1:
    # Nếu đã có kết quả SVG thì hiển thị
    if st.session_state.svg_result:
        st.subheader("🖼️ SVG Result")
        # Dùng st.image để hiển thị SVG, nó hỗ trợ zoom tự nhiên
        st.image(st.session_state.svg_result, use_container_width=True, caption="Generated SVG (Click to zoom)")

        # Nút để upload file mới
        if st.button("Upload another image"):
            st.session_state.svg_result = None
            st.session_state.uploaded_file_key = str(int(time.time())) # Reset uploader
            st.rerun()

    # Nếu chưa, hiển thị khu vực upload
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #cccccc; border-radius: 10px; padding: 50px 20px; text-align: center;">
                <h3>Drag and drop an image <br>or <b>browse to upload.</b></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload your photo",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed",
            key=st.session_state.uploaded_file_key
        )

        if uploaded_file is not None:
            # Lưu file vào session state để xử lý
            st.session_state.uploaded_image_bytes = uploaded_file.getvalue()
            st.image(st.session_state.uploaded_image_bytes, caption="Your uploaded image")


# --- Cột phải: Các tham số xử lý ---
with col2:
    st.subheader("Processing Parameters:")
    if st.button("Reset default", type="secondary"):
        reset_params()

    # Widget key để đảm bảo các widget được cập nhật giá trị sau khi reset
    key = st.session_state.get('widget_key', 'default')

    # Nhóm tham số
    st.markdown("---")
    st.markdown("**I/O & Mode**")
    st.session_state.params['mode'] = st.selectbox(
        "Mode", ["Auto", "Color", "Mono"], index=["Auto", "Color", "Mono"].index(st.session_state.params['mode']), key=f"{key}_mode"
    )
    st.session_state.params['k'] = st.number_input("K:", min_value=2, max_value=32, value=st.session_state.params['k'], key=f"{key}_k")

    st.markdown("**Image Region / Pre-processing**")
    st.session_state.params['min_area'] = st.number_input("Min area:", min_value=1, value=st.session_state.params['min_area'], key=f"{key}_min_area")
    st.session_state.params['area_max_ratio'] = st.number_input("Area max ratio:", min_value=0.1, max_value=1.0, value=st.session_state.params['area_max_ratio'], step=0.01, key=f"{key}_area_max_ratio")
    st.session_state.params['super_sample'] = st.number_input("Super sample:", min_value=1, max_value=8, value=st.session_state.params['super_sample'], key=f"{key}_super_sample")
    st.session_state.params['decimate'] = st.number_input("Decimate:", min_value=1, max_value=10, value=st.session_state.params['decimate'], key=f"{key}_decimate")
    st.session_state.params['subpixel_sigma'] = st.number_input("Subpixel sigma:", min_value=0.1, max_value=5.0, value=st.session_state.params['subpixel_sigma'], step=0.1, key=f"{key}_subpixel_sigma")

    st.markdown("**\"Edge Repair\" & Morphology**")
    st.session_state.params['morph_close'] = st.number_input("Morph close:", min_value=0, max_value=20, value=st.session_state.params['morph_close'], key=f"{key}_morph_close")
    st.session_state.params['morph_open'] = st.number_input("Morph open:", min_value=0, max_value=20, value=st.session_state.params['morph_open'], key=f"{key}_morph_open")

    st.markdown("**Corners / Lines & Error Tolerance**")
    st.session_state.params['corner_eps'] = st.number_input("Corner eps:", min_value=0.1, max_value=5.0, value=st.session_state.params['corner_eps'], step=0.1, key=f"{key}_corner_eps")
    st.session_state.params['line_tol'] = st.number_input("Line tol:", min_value=0.01, max_value=1.0, value=st.session_state.params['line_tol'], step=0.01, key=f"{key}_line_tol")
    st.session_state.params['max_err'] = st.number_input("Max err:", min_value=0.1, max_value=5.0, value=st.session_state.params['max_err'], step=0.01, key=f"{key}_max_err")
    st.session_state.params['repair_clean'] = st.number_input("Repair clean:", min_value=0.01, max_value=5.0, value=st.session_state.params['repair_clean'], step=0.01, key=f"{key}_repair_clean")
    st.session_state.params['curve_sigma'] = st.number_input("Curve sigma:", min_value=0.0, max_value=5.0, value=st.session_state.params['curve_sigma'], step=0.1, key=f"{key}_curve_sigma")

    st.markdown("---")

    # Nút Process và Download
    buttons_col1, buttons_col2 = st.columns(2)

    with buttons_col1:
        # Nút Download bị vô hiệu hóa nếu chưa có kết quả
        download_disabled = st.session_state.svg_result is None
        st.download_button(
            label="Download",
            data=st.session_state.svg_result if st.session_state.svg_result else "",
            file_name="converted_image.svg",
            mime="image/svg+xml",
            disabled=download_disabled,
            use_container_width=True,
        )

    with buttons_col2:
        process_button = st.button("Process", type="primary", use_container_width=True)

# --- Logic xử lý ---
if process_button:
    if 'uploaded_image_bytes' not in st.session_state or st.session_state.uploaded_image_bytes is None:
        st.warning("Please upload an image first!")
    else:
        with st.spinner('Processing your image... This may take a moment.'):
            try:
                # --- THAY THẾ HOÀN TOÀN ĐOẠN CODE ĐỌC ẢNH ---

                # ĐOẠN CODE CŨ:
                # file_bytes = np.asarray(bytearray(st.session_state.uploaded_image_bytes), dtype=np.uint8)
                # img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # ĐOẠN CODE MỚI (ỔN ĐỊNH HƠN):
                # 1. Dùng Pillow để mở ảnh từ bytes (linh hoạt hơn)
                pil_image = Image.open(BytesIO(st.session_state.uploaded_image_bytes))
                
                # 2. Chuyển ảnh sang định dạng RGB (loại bỏ kênh alpha nếu có)
                pil_image_rgb = pil_image.convert('RGB')

                # 3. Chuyển ảnh Pillow RGB thành mảng NumPy
                rgb_image_np = np.array(pil_image_rgb)

                # 4. Chuyển từ định dạng RGB (Pillow) sang BGR (OpenCV)
                img_bgr = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)

                # ---------------------------------------------------

                if img_bgr is None:
                    st.error("❌ Không thể xử lý file ảnh. Vui lòng thử một file khác.")
                    st.stop()

                temp_output_path = "temp_output.svg"
                params = st.session_state.params
                
                # Gọi hàm xử lý chính với img_bgr đã được chuẩn hóa
                convert_png_to_svg(
                    input_path=None,
                    output_path=temp_output_path,
                    img_bgr_in=img_bgr, # Truyền ảnh đã được xử lý vào đây
                    mode=params['mode'].lower(),
                    k_colors=params['k'],
                    min_area_px=params['min_area'],
                    area_max_ratio=params['area_max_ratio'],
                    super_sample=params['super_sample'],
                    decimate=params['decimate'],
                    subpixel_sigma=params['subpixel_sigma'],
                    repair_clean=params['repair_clean'],
                    corner_eps=params['corner_eps'],
                    line_tol=params['line_tol'],
                    max_err=params['max_err'],
                    morph_close=params['morph_close'],
                    morph_open=params['morph_open'],
                    curve_sigma=params['curve_sigma']
                )

                with open(temp_output_path, "r", encoding="utf-8") as f:
                    svg_content = f.read()

                st.session_state.svg_result = svg_content
                st.success("✅ Conversion successful!")
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

