import streamlit as st
import numpy as np
import cv2 as cv
import joblib
from PIL import Image
import time
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Đọc và tiền xử lý dữ liệu",
    page_icon="🌐",
)

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'full_data' not in st.session_state:
    st.session_state['full_data'] = None
if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = None
if 'normalization_option' not in st.session_state:
    st.session_state['normalization_option'] = "Không chuẩn hóa"  # Giá trị mặc định

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
    background-image: url("https://drive.google.com/file/d/1HsEqY3G5e-jMDVjlxqD51DBHg-_vbD2I/view?usp=sharing");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

st.title("Đọc và tiền xử lý dữ liệu")

uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu ", type=['csv'])


if uploaded_file is not None:
    if st.session_state['uploaded_file'] != uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        try:
            maindf = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            st.session_state['full_data'] = maindf
        except Exception as e:
            st.error(f"Có lỗi khi đọc dữ liệu: {str(e)}")

# Đọc và tiền xử lý dữ liệu
st.header("1. Đọc và kiểm tra dữ liệu")

try:
    if st.session_state['full_data'] is not None:
        maindf = st.session_state['full_data']
        
        # 2.1 Thông tin cơ bản
        st.markdown("#### 2.1. Thông tin cơ bản về dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Số ngày trong dataset: {maindf.shape[0]}")
            st.write(f"- Số trường dữ liệu: {maindf.shape[1]}")
        with col2:
            st.write(f"- Ngày bắt đầu: {maindf.index.min().strftime('%Y-%m-%d')}")
            st.write(f"- Ngày kết thúc: {maindf.index.max().strftime('%Y-%m-%d')}")

        # 2.2 Dữ liệu mẫu và biểu đồ giá
        st.markdown("#### 2.2. Dữ liệu mẫu và biểu đồ giá")
        show_sample = st.checkbox("Xem dữ liệu mẫu")
        if show_sample:
            st.dataframe(maindf.head())
        
        # Chọn cột để hiển thị
        selected_column = st.selectbox(
            "Chọn cột để hiển thị:",
            ['Open', 'High', 'Low', 'Close', 'Tất cả']
        )

        # Thêm radio button để chọn giữa vẽ biểu đồ hoặc xem số liệu
        view_data_or_chart = st.radio("Chọn hiển thị cho cột đã chọn:", ("Vẽ biểu đồ", "Xem số liệu"))

        if view_data_or_chart == "Vẽ biểu đồ":
            # Vẽ biểu đồ cho từng cột riêng biệt
            if selected_column == 'Tất cả':  # Nếu chọn 'Tất cả', vẽ tất cả các cột
                for col in ['Open', 'High', 'Low', 'Close']:
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.plot(maindf.index, maindf[col], linewidth=2, label=col)
                    ax.set_title(f'Historical {col} Price Chart', pad=10)
                    ax.set_xlabel('Date', fontsize=12)  # Chú thích cho trục x
                    ax.set_ylabel(f'{col} Price', fontsize=12)  # Chú thích cho trục y
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend()  # Hiển thị chú thích cho đường biểu đồ
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(maindf.index, maindf[selected_column], linewidth=2, label=selected_column)
                ax.set_title(f'Historical {selected_column} Price Chart', pad=10)
                ax.set_xlabel('Date', fontsize=12)  # Chú thích cho trục x
                ax.set_ylabel(f'{selected_column} Price', fontsize=12)  # Chú thích cho trục y
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='x', rotation=45)
                ax.legend()  # Hiển thị chú thích cho đường biểu đồ
                plt.tight_layout()
                st.pyplot(fig)

            # # Hiển thị bảng dữ liệu cho cột được chọn
            # st.markdown(f"#### Dữ liệu cho cột {selected_column} cùng với Date")
            # st.dataframe(maindf[[selected_column]].reset_index())

        elif view_data_or_chart == "Xem số liệu":
            # Hiển thị số liệu cho từng cột riêng biệt
            if selected_column == 'Tất cả':
                for col in ['Open', 'High', 'Low', 'Close']:
                    st.markdown(f"#### Dữ liệu cho cột {col} cùng với Date")
                    st.dataframe(maindf[[col]].reset_index())
            else:
                st.markdown(f"#### Dữ liệu cho cột {selected_column} cùng với Date")
                st.dataframe(maindf[[selected_column]].reset_index())

        elif view_data_or_chart == "Xem số liệu":
            selected_column = st.selectbox(
                "Chọn cột để xem số liệu:",
                ['Open', 'High', 'Low', 'Close']
            )
            st.markdown(f"#### Dữ liệu cho cột {selected_column} cùng với Date")
            st.dataframe(maindf[[selected_column]].reset_index())

        # 2.3 Thống kê mô tả
        st.markdown("#### 2.3. Thống kê mô tả")
        if st.checkbox("Xem thống kê cơ bản"):
            st.write(maindf.describe())

        # Định nghĩa các hàm chuẩn hóa
        def min_max_normalize(df):
            """Chuẩn hóa DataFrame bằng phương pháp chuẩn hóa Min-Max."""
            return (df - df.min()) / (df.max() - df.min())

        def z_score_normalize(df):
            """Chuẩn hóa DataFrame bằng phương pháp chuẩn hóa Z-score."""
            return (df - df.mean()) / df.std()

        # Thêm tùy chọn chuẩn hóa
        st.header("Chọn phương pháp chuẩn hóa dữ liệu")
        st.session_state['normalization_option'] = st.selectbox(
            "Chọn phương pháp chuẩn hóa:",
            options=["Không chuẩn hóa", "Chuẩn hóa Min-Max", "Chuẩn hóa Z-score"],
            index=["Không chuẩn hóa", "Chuẩn hóa Min-Max", "Chuẩn hóa Z-score"].index(st.session_state['normalization_option'])  # Giữ nguyên giá trị đã chọn
        )

        # Lưu dữ liệu gốc để vẽ biểu đồ sau khi chuẩn hóa
        original_data = maindf.copy()

        # Kiểm tra xem dữ liệu đã được chuẩn hóa chưa
        if 'normalized_data' not in st.session_state:
            st.session_state['normalized_data'] = maindf.copy()  # Lưu dữ liệu gốc


        # Áp dụng chuẩn hóa dựa trên lựa chọn của người dùng
        if st.session_state['normalization_option'] == "Chuẩn hóa Min-Max":
            st.session_state['normalized_data'] = min_max_normalize(maindf)
            st.success("Dữ liệu đã được chuẩn hóa bằng phương pháp Min-Max.")
        elif st.session_state['normalization_option'] == "Chuẩn hóa Z-score":
            st.session_state['normalized_data'] = z_score_normalize(maindf)
            st.success("Dữ liệu đã được chuẩn hóa bằng phương pháp Z-score.")
        else:
            st.session_state['normalized_data'] = maindf.copy()  # Không chuẩn hóa
            st.success("Dữ liệu không được chuẩn hóa.")
            
        # Lưu dữ liệu đã chuẩn hóa vào session state
        st.session_state['full_data'] = st.session_state['normalized_data']
        
         # Vẽ biểu đồ trước và sau khi chuẩn hóa
        if st.session_state['normalization_option'] != "Không chuẩn hóa":
            st.subheader("Biểu đồ trước và sau khi chuẩn hóa")
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))

            # Biểu đồ dữ liệu gốc
            ax[0].plot(original_data.index, original_data['Open'], label='Open', color='blue')
            ax[0].plot(original_data.index, original_data['High'], label='High', color='orange')
            ax[0].plot(original_data.index, original_data['Low'], label='Low', color='green')
            ax[0].plot(original_data.index, original_data['Close'], label='Close', color='red')
            ax[0].set_title('Dữ liệu gốc', pad=10)
            ax[0].set_xlabel('Date')
            ax[0].set_ylabel('Giá')
            ax[0].legend()
            ax[0].grid(True)

            # Biểu đồ dữ liệu đã chuẩn hóa
            ax[1].plot(st.session_state['normalized_data'].index, st.session_state['normalized_data']['Open'], label='Open', color='blue')
            ax[1].plot(st.session_state['normalized_data'].index, st.session_state['normalized_data']['High'], label='High', color='orange')
            ax[1].plot(st.session_state['normalized_data'].index, st.session_state['normalized_data']['Low'], label='Low', color='green')
            ax[1].plot(st.session_state['normalized_data'].index, st.session_state['normalized_data']['Close'], label='Close', color='red')
            ax[1].set_title('Dữ liệu đã chuẩn hóa', pad=10)
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Giá (đã chuẩn hóa)')
            ax[1].legend()
            ax[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

         # 2.4 Kiểm tra và xử lý dữ liệu thiếu
        st.markdown("#### 2.4. Kiểm tra và xử lý dữ liệu thiếu")
        missing_values = maindf.isnull().sum()

        if missing_values.sum() > 0:
            st.write("Số lượng giá trị thiếu trong từng trường:")
            st.write(missing_values)
            
            if st.checkbox("Xử lý dữ liệu thiếu"):
                method = st.selectbox(
                    "Chọn phương pháp xử lý:",
                    ["Giá trị trung bình", "Xóa dòng", "Giá trị trung bình của 5 giá trị gần nhất"]
                )
                
                if st.button("Áp dụng xử lý"):
                    if method == "Giá trị trung bình của cột":
                        maindf = maindf.fillna(maindf.mean())
                    elif method == "Giá trị trung bình của 5 giá trị gần nhất":
                        maindf = maindf.fillna(maindf.rolling(window=5, center=True, min_periods=1).mean())
                    else:  # Xóa dòng
                        maindf = maindf.dropna()
                    
                    # Lấy 365 ngày gần nhất sau khi xử lý
                    maindf = maindf.tail(365)
                    st.write(f"Đã lọc lại còn {len(maindf)} mẫu dữ liệu gần nhất để thực hiện tiếp")

                    missing_values_after = maindf.isnull().sum()
                    if missing_values_after.sum() == 0:
                        st.success("✅ Đã xử lý xong tất cả giá trị thiếu")
                    else:
                        st.write("Số lượng giá trị thiếu sau khi xử lý:")
                        st.write(missing_values_after)
        else:
            st.success("✅ Dataset không có giá trị thiếu")
            # Thêm dòng này để lấy 365 mẫu gần nhất khi không có giá trị thiếu
            maindf = maindf.tail(365)
            st.write(f"Đã lọc lại còn {len(maindf)} mẫu dữ liệu gần nhất để thực hiện tiếp")
        
        st.session_state['full_data'] = maindf
        st.session_state['filtered_data'] = maindf[['Open', 'High', 'Low', 'Close']]

except Exception as e:
    st.error(f"Có lỗi khi xử lý dữ liệu: {str(e)}")