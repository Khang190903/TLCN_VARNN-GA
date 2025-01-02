import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="🏚️",
)
logo = Image.open('images\cntt.png')
st.image(logo, width=1000)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/blue-copy-space-digital-background_23-2148821698.jpg");
    background-size: 100% 100%;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://img.freepik.com/free-vector/blue-copy-space-digital-background_23-2148821698.jpg");
    background-position: center;
    border: 2px solid #4CAF50; /* Đặt khung cho sidebar */
    border-radius: 10px; /* Bo góc khung */
    padding: 10px; /* Khoảng cách bên trong khung */
}
[data-testid="stSidebarNav"] button {
    border: 1px solid #4CAF50; /* Đặt khung cho các nút */
    border-radius: 5px; /* Bo góc khung nút */
    margin: 5px 0; /* Khoảng cách giữa các nút */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# logo_path = "./VCT.png"
# st.sidebar.image(logo_path, width=200)

st.write("## 🌐TIỂU LUẬN CHUYÊN NGÀNH")
st.write("## ĐỀ TÀI: TÌM HIỂU MẠNG NEURON VECTOR TỰ HỒI QUY KẾT HỢP VỚI THUẬT TOÁN DI TRUYỀN DÙNG TRONG DỰ BÁO ")
st.write("Giảng viên hướng dẫn: TS. Nguyễn Thành Sơn")
st.write("🌐THÔNG TIN SINH VIÊN:")
st.write("21110891 - Đỗ Huỳnh Gia Khang")
st.write("21110608 - Trần Quốc Phúc")


st.markdown(
    """
    ## 🌐CÁC BƯỚC THỰC HIỆN:
    🌐Đồ án gồm 07 bước thực hiện:
    - 1️⃣ Đọc và tiền xử lý dữ liệu
    - 2️⃣ Kiểm tra chuỗi dừng, phương sai 
    - 3️⃣ Chia tập huấn luyện, validation và kiểm tra
    - 4️⃣ Xác định LAG
    - 5️⃣ Tối ưu hóa siêu tham số bằng Genetic Algorithm(GA)
    - 6️⃣ Huấn luyện mô hình VARNN
    - 7️⃣ Đánh giá mô hình
    """
)

