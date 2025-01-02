import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Chia tập huấn luyện, validation và kiểm tra",
    page_icon="🌐",
)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
  background-image: url("");

    
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

st.title("Phân chia các tập")

# Kiểm tra dữ liệu
if 'filtered_data' not in st.session_state:
    st.warning("⚠️ Vui lòng xử lý dữ liệu ở các trang trước!")
    st.stop()

# Lấy dữ liệu và biến đổi log difference
data = st.session_state['filtered_data']
df_log = np.log(data + 1)
df_log_diff = df_log.diff().dropna()

# Phần cài đặt tỷ lệ chia
st.header("1. Thiết lập tỷ lệ chia dữ liệu")

# Khởi tạo giá trị trong session_state chỉ khi chưa tồn tại
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['train_ratio'] = 80  # Tỷ lệ tập huấn luyện
    st.session_state['valid_ratio'] = 20  # Tỷ lệ validation
    st.session_state['test_ratio'] = 0    # Tỷ lệ tập kiểm tra (mặc định là 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.session_state['train_ratio'] = st.slider(
        "Tỷ lệ tập huấn luyện tổng (%)", 
        min_value=60, 
        max_value=90, 
        value=st.session_state['train_ratio'],
        step=5
    )
with col2:
    st.session_state['valid_ratio'] = st.slider(
        "Tỷ lệ validation từ tập huấn luyện (%)", 
        min_value=10, 
        max_value=30, 
        value=st.session_state['valid_ratio'],
        step=5
    )
with col3:
    # Tính toán tỷ lệ test dựa trên tỷ lệ train và valid
    st.session_state['test_ratio'] = 100 - st.session_state['train_ratio'] 
    st.metric("Tỷ lệ tập kiểm tra (%)", st.session_state['test_ratio'])  # Hiển thị tỷ lệ kiểm tra

# Sử dụng giá trị từ session_state
train_ratio = st.session_state['train_ratio']
valid_ratio = st.session_state['valid_ratio']
test_ratio = st.session_state['test_ratio']

# Tính toán các tỷ lệ
TRAIN_RATIO = train_ratio / 100
VALID_RATIO = valid_ratio / 100
TEST_RATIO = test_ratio / 100

# Chia dữ liệu
total_samples = len(df_log_diff)
train_size = int(total_samples * TRAIN_RATIO)
test_size = total_samples - train_size

train_data_full = df_log_diff[:train_size]
test_data = df_log_diff[train_size:]

# Chia train_full thành train và validation
valid_size = int(len(train_data_full) * VALID_RATIO)
train_data = train_data_full[:-valid_size]
valid_data = train_data_full[-valid_size:]

# Hiển thị thông tin về tập dữ liệu
st.subheader("Thông tin về tập dữ liệu")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tập huấn luyện", 
              f"{len(train_data)} mẫu",
              f"{len(train_data)/total_samples*100:.1f}%")
with col2:
    st.metric("Tập validation", 
              f"{len(valid_data)} mẫu",
              f"{len(valid_data)/total_samples*100:.1f}%")
with col3:
    st.metric("Tập kiểm tra", 
              f"{len(test_data)} mẫu",
              f"{len(test_data)/total_samples*100:.1f}%")

# Hiển thị thông tin về thời gian
st.subheader("Khoảng thời gian của các tập")
col1, col2 = st.columns(2)
with col1:
    st.write("**Tập huấn luyện:**")
    st.write(f"- Từ: {train_data.index[0].strftime('%Y-%m-%d')}")
    st.write(f"- Đến: {train_data.index[-1].strftime('%Y-%m-%d')}")
    
    st.write("**Tập validation:**")
    st.write(f"- Từ: {valid_data.index[0].strftime('%Y-%m-%d')}")
    st.write(f"- Đến: {valid_data.index[-1].strftime('%Y-%m-%d')}")

with col2:
    st.write("**Tập kiểm tra:**")
    st.write(f"- Từ: {test_data.index[0].strftime('%Y-%m-%d')}")
    st.write(f"- Đến: {test_data.index[-1].strftime('%Y-%m-%d')}")

# Vẽ biểu đồ phân chia
st.subheader("Biểu đồ phân chia dữ liệu")
fig, ax = plt.subplots(figsize=(12, 6))

# Vẽ dữ liệu train
ax.plot(train_data.index, 
        train_data['Close'], 
        label='Train', 
        color='blue',
        alpha=0.7)

# Vẽ dữ liệu validation
ax.plot(valid_data.index, 
        valid_data['Close'], 
        label='Validation', 
        color='green',
        alpha=0.7)

# Vẽ dữ liệu test
ax.plot(test_data.index, 
        test_data['Close'], 
        label='Test', 
        color='red',
        alpha=0.7)

ax.set_title('Phân chia tập dữ liệu (sau biến đổi log difference)')
ax.set_xlabel('Thời gian')
ax.set_ylabel('Giá đóng cửa (Close)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Lưu dữ liệu đã chia vào session state
st.session_state['train_data'] = train_data
st.session_state['valid_data'] = valid_data
st.session_state['test_data'] = test_data
st.session_state['df_log_diff'] = df_log_diff