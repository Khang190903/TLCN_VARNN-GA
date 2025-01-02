import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant
st.set_page_config(
    page_title="Kiểm tra chuỗi dừng, phương sai",
    page_icon="🌐",
)
# Đặt nền cho trang Streamlit
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

st.title("Kiểm tra chuỗi dừng và phương sai")


def check_stationarity(data):
    # Thêm tùy chọn cho người dùng chọn cột
    selected_columns = st.multiselect("Chọn cột để kiểm tra chuỗi dừng:", data.columns.tolist(), default=data.columns.tolist())
    for column in selected_columns:
        with st.expander(f"🔍 Kiểm tra ADF cho {column}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                result = adfuller(data[column])
                
                # Thêm thông tin chi tiết hơn
                st.markdown("#### Kết quả kiểm định ADF")
                st.metric("ADF Statistic", f"{result[0]:.4f}")
                st.metric("p-value", f"{result[1]:.4f}")
                st.metric("Số quan sát", result[3])
                
                st.markdown("##### Critical values:")
                for key, value in result[4].items():
                    st.info(f"{key}: {value:.4f}")
                
                if result[1] > 0.05:
                    st.error("❌ KHÔNG là chuỗi dừng (p-value > 0.05)")
                    st.caption("Mean và std có thể thay đổi theo thời gian")
                else:
                    st.success("✅ Là chuỗi dừng (p-value < 0.05)")
                    st.caption("Mean và std tương đối ổn định")
            
            with col2:
                # Thêm biểu đồ rolling statistics
                fig, ax = plt.subplots(figsize=(8, 4))
                window_size = 30
                rolling_mean = data[column].rolling(window=window_size, min_periods=1, center=True).mean()
                rolling_std = data[column].rolling(window=window_size, min_periods=1, center=True).std()
                
                rolling_mean = rolling_mean.ewm(span=20, adjust=False).mean()
                rolling_std = rolling_std.ewm(span=20, adjust=False).mean()
                
                ax.plot(data[column], label='Original', alpha=0.5)
                ax.plot(rolling_mean, label='Rolling Mean', color='red')
                ax.plot(rolling_std, label='Rolling Std', color='green')
                ax.set_title(f'Rolling Statistics - {column}')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

# def check_heteroscedasticity(data):
#     st.subheader("📊 Kiểm tra phương sai không đổi")
    
#     for column in data.columns:
#         with st.container():
#             residuals = data[column] - data[column].mean()
#             exog = add_constant(data)
#             bp_test = het_breuschpagan(residuals, exog)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Breusch-Pagan Statistic", f"{bp_test[0]:.4f}")
#             with col2:
#                 st.metric("p-value", f"{bp_test[1]:.4f}")
            
#             if bp_test[1] > 0.05:
#                 st.success(f"✅ {column} có phương sai không đổi")
#             else:
#                 st.error(f"❌ {column} có phương sai thay đổi")
#             st.divider()
def check_heteroscedasticity(data):
    st.subheader("Kiểm tra phương sai không đổi")
    # Thêm tùy chọn cho người dùng chọn cột
    selected_columns = st.multiselect("Chọn cột để kiểm tra phương sai không đổi:", data.columns.tolist(), default=data.columns.tolist())
    for column in selected_columns:
        with st.expander(f"Phân tích phương sai cho {column}", expanded=True):
            # Tính returns theo log để giảm thiểu ảnh hưởng của outliers
            log_returns = np.log(data[column] / data[column].shift(1)).dropna()
            rolling_var = log_returns.rolling(window=30).var()  # Tăng window size lên 30
            
            # Vẽ biểu đồ
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Biểu đồ giá (đã log transform)
            ax1.plot(np.log(data[column]), label=f'Log {column}')
            ax1.set_title(f'Log giá {column} theo thời gian')
            ax1.set_xlabel('Thời gian')
            ax1.set_ylabel('Log giá')
            ax1.legend()
            ax1.grid(True)
            
            # Biểu đồ phương sai
            ax2.plot(rolling_var, color='red', label='Phương sai động')
            ax2.set_title('Phương sai động (cửa sổ 30 ngày)')
            ax2.set_xlabel('Thời gian')
            ax2.set_ylabel('Phương sai')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            try:
                # Cải thiện kiểm định Breusch-Pagan
                # Sử dụng squared log returns làm biến phụ thuộc
                squared_returns = log_returns**2
                time_trend = np.arange(len(squared_returns))
                exog = add_constant(time_trend)
                bp_test = het_breuschpagan(squared_returns, exog)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Breusch-Pagan Statistic", f"{bp_test[0]:.4f}")
                with col2:
                    st.metric("p-value", f"{bp_test[1]:.4f}")
                
                # Thêm mức ý nghĩa 1%
                if bp_test[1] > 0.01:  # Thay đổi từ 0.05 thành 0.01
                    st.success(f"{column}: Phương sai tương đối đồng nhất")
                    st.caption("Phương sai của chuỗi không có xu hướng biến động mạnh")
                else:
                    st.error(f"{column}: Phương sai không đồng nhất")
                    st.caption("Phương sai của chuỗi có xu hướng biến động theo thời gian")
                
                # Thống kê mô tả
                st.markdown("Thống kê mô tả về biến động")
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Phương sai trung bình", f"{log_returns.var():.6f}")
                with col4:
                    st.metric("Biến động lớn nhất", f"{log_returns.max():.6f}")
                with col5:
                    st.metric("Biến động nhỏ nhất", f"{log_returns.min():.6f}")
                
            except Exception as e:
                st.error(f"Lỗi khi phân tích {column}: {str(e)}")
            
            st.divider()

def check_stationarity_after_diff(data):
    st.subheader("🔄 Kiểm tra sau differencing")
    # Thêm tùy chọn cho người dùng chọn cột
    selected_columns = st.multiselect("Chọn cột để kiểm tra sau differencing:", data.columns.tolist(), default=data.columns.tolist())
    for column in selected_columns:
        series = data[column]
        order = 1
        max_order = 3
        
        with st.expander(f"Phân tích {column}", expanded=True):
            while order <= max_order:
                result = adfuller(series)
                
                st.markdown(f"#### Lần differencing thứ {order}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ADF Statistic", f"{result[0]:.4f}")
                with col2:
                    st.metric("p-value", f"{result[1]:.4f}")
                
                st.caption("Critical values:")
                for key, value in result[4].items():
                    st.text(f"{key}: {value:.4f}")
                
                if result[1] < 0.05:
                    st.success(f"✅ Chuỗi đã dừng sau {order} lần differencing")
                    break
                
                order += 1
                series = series.diff().dropna()
                
                if order > max_order:
                    st.warning(f"⚠️ Chuỗi không đạt tính dừng sau {max_order} lần differencing")


def check_data_stationarity_and_heteroscedasticity(data):
    st.subheader("📈 Phân tích chi tiết")
    # Thêm tùy chọn cho người dùng chọn cột
    selected_columns = st.multiselect("Chọn cột để phân tích chi tiết:", data.columns.tolist(), default=data.columns.tolist())
    for column in selected_columns:
        with st.expander(f"Phân tích {column}", expanded=True):
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            window_size = 30
            rolling_mean = data[column].rolling(window=window_size, min_periods=1, center=True).mean()
            rolling_std = data[column].rolling(window=window_size, min_periods=1, center=True).std()
            
            rolling_mean = rolling_mean.ewm(span=20, adjust=False).mean()
            rolling_std = rolling_std.ewm(span=20, adjust=False).mean()
            
            ax.plot(data[column], label='Original', alpha=0.5, color='blue')
            ax.plot(rolling_mean, label='Rolling Mean', color='red', linewidth=2)
            ax.plot(rolling_std, label='Rolling Std', color='green', linewidth=2)
            ax.set_title(f'Phân tích tính dừng cho {column}')
            ax.set_xlabel('Thời gian')
            ax.set_ylabel('Giá trị')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Hiển thị thống kê
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{data[column].mean():.4f}")
            with col2:
                st.metric("Standard Deviation", f"{data[column].std():.4f}")
            
            # Kiểm tra ADF
            result = adfuller(data[column])
            st.markdown("#### Kết quả kiểm định ADF")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("ADF Statistic", f"{result[0]:.4f}")
            with col4:
                st.metric("p-value", f"{result[1]:.4f}")
            
            st.markdown("##### Critical values:")
            for key, value in result[4].items():
                st.info(f"{key}: {value:.4f}")
            
            if result[1] > 0.05:
                st.error("❌ KHÔNG là chuỗi dừng (p-value > 0.05)")
                st.caption("Mean và std có thể thay đổi theo thời gian")
            else:
                st.success("✅ Là chuỗi dừng (p-value < 0.05)")
                st.caption("Mean và std tương đối ổn định")

# Kiểm tra xem có dữ liệu trong session state không
if 'full_data' not in st.session_state:
    st.warning("⚠️ Vui lòng đọc dữ liệu ở trang 'Đọc và tiền xử lý dữ liệu' trước!")
    st.stop()

# Lấy dữ liệu từ session state
data = st.session_state['filtered_data'][['Open', 'High', 'Low', 'Close']]

# Hiển thị thông tin về toàn bộ dữ liệu
st.subheader("📊 Thông tin dữ liệu")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng số ngày", len(data))
with col2:
    st.metric("Từ", data.index.min().strftime('%Y-%m-%d'))
with col3:
    st.metric("Đến", data.index.max().strftime('%Y-%m-%d'))

# Hiển thị preview dữ liệu
with st.expander("Xem dữ liệu mẫu"):
    st.dataframe(data.head())


# Tạo tabs để phân chia các phân tích
tab1, tab2, tab3, tab4 = st.tabs([
    "Kiểm tra ADF", 
    "Kiểm tra phương sai", 
    "Kiểm tra sau differencing",
    "Phân tích chi tiết"
])

with tab1:
   # Biến đổi log và differencing
    df_log = np.log(data + 1)
    df_log_diff = df_log.diff().dropna()
    check_stationarity(data)
    
with tab2:
    check_heteroscedasticity(data)
    
with tab3:
    # check_stationarity_after_diff(data)
    check_stationarity_after_diff(df_log_diff)
    
with tab4:
    check_data_stationarity_and_heteroscedasticity(df_log_diff)
            
