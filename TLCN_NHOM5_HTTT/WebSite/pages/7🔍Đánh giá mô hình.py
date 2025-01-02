import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time  # Nhập mô-đun time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_page_config(page_title="Đánh giá mô hình", page_icon="🔍")

# Thiết lập background
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
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Đánh giá mô hình")

# Kiểm tra dữ liệu
if not all(key in st.session_state for key in ['model_optimized', 'model_default', 'X_test', 'y_test']):
    st.warning("⚠️ Vui lòng thực hiện các bước trước đó!")
    st.stop()


st.header("1. Đánh giá mô hình đã huấn luyện trên tập test")

# Khởi tạo các biến để tránh lỗi khi mô hình chưa được huấn luyện
mse_test = rmse_test = mae_test = cv_rmse_test = None

# Kiểm tra xem mô hình đã huấn luyện chưa
if 'model_optimized' in st.session_state:
    # Dự đoán trên tập test
    predictions_optimized = st.session_state['model_optimized'].predict(st.session_state['X_test'])  # Sửa đổi để truy cập từ session_state
            
    # Chuyển đổi ngược về dạng gốc
    y_test_original = np.exp(np.cumsum(st.session_state['y_test'], axis=0)) - 1
    predictions_original = np.exp(np.cumsum(predictions_optimized, axis=0)) - 1

    # Tính toán các chỉ số
    mse_test = mean_squared_error(y_test_original, predictions_original)  # Sử dụng dữ liệu gốc
    rmse_test = math.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_original, predictions_original)
    
    # Tính CV(RMSE) trên dữ liệu gốc
    mean_actual = np.mean(y_test_original)  # Tính giá trị trung bình của giá trị thực tế gốc
    cv_rmse_test = (rmse_test / mean_actual) * 100 if mean_actual != 0 else 0  # Tính CV(RMSE)

    # Hiển thị các chỉ số
    col1, col2, col3, col4 = st.columns(4)  # Tạo 4 cột để hiển thị các chỉ số
    with col1:
        st.metric("MSE (Test)", f"{mse_test:.6f}")  # Hiển thị MSE
    with col2:
        st.metric("RMSE (Test)", f"{rmse_test:.6f}")  # Hiển thị RMSE
    with col3:
        st.metric("MAE (Test)", f"{mae_test:.6f}")  # Hiển thị MAE
    with col4:
        st.metric("CV(RMSE) (Test)", f"{cv_rmse_test:.2f}%")  # Hiển thị CV(RMSE)

    # Tạo menu lựa chọn cho người dùng
    selected_feature = st.selectbox("Chọn đặc trưng để so sánh:", ['Open', 'High', 'Low', 'Close'])

    # Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán cho đặc trưng đã chọn
    st.subheader(f"Biểu đồ so sánh cho {selected_feature}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state['y_test'][:, ['Open', 'High', 'Low', 'Close'].index(selected_feature)], 
            label='Giá trị thực tế', color='blue')  # Vẽ giá trị thực tế
    ax.plot(predictions_optimized[:, ['Open', 'High', 'Low', 'Close'].index(selected_feature)], 
            label='Giá trị dự đoán', color='red', linestyle='--')  # Sửa đổi ở đây
    ax.set_title(f'So sánh giá trị thực tế và giá trị dự đoán cho {selected_feature}')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Giá trị')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)  # Hiển thị biểu đồ
else:
    st.warning("⚠️ Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình trước khi đánh giá.")

# Đưa ra kết luận dựa trên các chỉ số
if mse_test is not None and rmse_test is not None and mae_test is not None and cv_rmse_test is not None:
    st.subheader("Kết luận đánh giá mô hình")

    # Đánh giá độ chính xác
    if mse_test < 1:  # Giả sử ngưỡng 1 cho MSE
        st.success("✅ Mô hình có độ chính xác cao với MSE thấp.")
    else:
        st.warning("⚠️ Mô hình có thể cần cải thiện với MSE cao.")

    if rmse_test < 1:  # Giả sử ngưỡng 1 cho RMSE
        st.success("✅ Mô hình có độ chính xác cao với RMSE thấp.")
    else:
        st.warning("⚠️ Mô hình có thể cần cải thiện với RMSE cao.")

    if mae_test < 1:  # Giả sử ngưỡng 1 cho MAE
        st.success("✅ Mô hình có độ chính xác cao với MAE thấp.")
    else:
        st.warning("⚠️ Mô hình có thể cần cải thiện với MAE cao.")

    # Đánh giá độ biến động
    if cv_rmse_test < 10:  # Giả sử ngưỡng 10% cho CV(RMSE)
        st.success("✅ Mô hình có độ ổn định cao với CV(RMSE) thấp.")
    else:
        st.info("ℹ️ Mô hình có thể không ổn định với CV(RMSE) cao.")

    # Tổng hợp kết quả
    if mse_test < 1 and rmse_test < 1 and mae_test < 1 and cv_rmse_test < 10:
        st.success("🎉 Mô hình đạt hiệu suất tuyệt vời với tất cả các chỉ số đều ở mức rất thấp, chứng tỏ độ chính xác và ổn định cao.")
    elif mse_test < 1 and rmse_test < 1 and mae_test < 1 and cv_rmse_test > 10:  # Điều kiện mới
        st.success("🎉 Mô hình có độ chính xác tốt, với các chỉ số MSE, RMSE và MAE đều rất thấp. Tuy nhiên, chỉ số CV(RMSE) cao hơn ngưỡng yêu cầu, cho thấy cần điều chỉnh để cải thiện tính ổn định.")
    else:
        st.info("ℹ️ Mô hình có tiềm năng nhưng cần được cải thiện để đạt hiệu suất dự báo cao hơn.")







# 2. Dự đoán giá Bitcoin
st.header("2. Dự đoán giá")

@st.cache_data
def inverse_transform_predictions(predictions, original_data):
    """Hàm chuyển đổi dự đoán về dạng gốc"""
    predictions_exp = np.exp(predictions) - 1
    predictions_restored = {}
    columns = ['Open', 'High', 'Low', 'Close']
    
    for i, col in enumerate(columns):
        last_value = original_data[col].iloc[-1]
        cumulative_changes = np.cumsum(predictions_exp[:, i])
        predictions_restored[col] = last_value * (1 + cumulative_changes)
        if np.any(predictions_restored[col] < 0):
           predictions_restored[col] = np.maximum(predictions_restored[col], 0)
    
    predictions_df = pd.DataFrame(predictions_restored)
    last_date = original_data.index[-1]
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                            periods=len(predictions_df), freq='D')
    predictions_df.index = new_dates
    
    predictions_df['High'] = np.maximum(predictions_df[['Open', 'High', 'Close']].max(axis=1),
                                      predictions_df['High'])
    predictions_df['Low'] = np.minimum(predictions_df[['Open', 'Low', 'Close']].min(axis=1),
                                     predictions_df['Low'])
    
    return predictions_df


@st.cache_data
def rolling_predict(model, last_window, n_steps):
    """Hàm dự đoán rolling forecast"""
    predictions = []
    current_window = last_window.copy()
    
    for _ in range(n_steps):
        # Reshape window để phù hợp với input của mô hình (1, lag_order, features)
        model_input = current_window.reshape(1, current_window.shape[0], current_window.shape[1])
        
        # Dự đoán một bước tiếp theo
        next_pred = model.predict(model_input, verbose=0)
        predictions.append(next_pred[0])
        
        # Cập nhật cửa sổ dự đoán bằng cách loại bỏ giá trị đầu tiên và thêm dự đoán mới
        current_window = np.vstack([current_window[1:], next_pred])
    
    return np.array(predictions)



# Lấy cửa sổ cuối cùng từ X_test với kích thước đúng với lag_order
last_window = st.session_state['X_test'][-1].reshape(st.session_state['best_lag'], -1)

# Số ngày muốn dự đoán
n_future = st.slider("Chọn số ngày muốn dự đoán:", 
                    min_value=5, 
                    max_value=30, 
                    value=10)

# Thực hiện dự đoán rolling
future_predictions = rolling_predict(st.session_state['model_optimized'], 
                                   last_window, 
                                   n_future)

# Chuyển đổi dự đoán về dạng gốc
predictions_restored = inverse_transform_predictions(future_predictions, st.session_state['filtered_data'])


# Hiển thị kết quả dự đoán
st.subheader("2.1. Bảng giá dự đoán")
st.dataframe(predictions_restored.head(10))


# Chọn đặc trưng để hiển thị trong biểu đồ dự đoán
feature_options = ['Open', 'High', 'Low', 'Close', 'All']
selected_feature = st.selectbox("Chọn đặc trưng để hiển thị trong biểu đồ dự đoán:", feature_options)

# Vẽ biểu đồ dự đoán
st.subheader("2.2. Biểu đồ dự đoán")
fig = go.Figure()

# Nếu người dùng chọn 'All', vẽ tất cả các đặc trưng
if selected_feature == 'All':
    for feature in ['Open', 'High', 'Low', 'Close']:
        # Thêm dữ liệu thực tế
        fig.add_trace(go.Scatter(x=st.session_state['filtered_data'].index[-n_future:],  # Sử dụng n_future để lấy đúng số ngày
                                  y=st.session_state['filtered_data'][feature][-n_future:],  # Sử dụng n_future để lấy đúng số ngày
                                  name=f'Actual {feature}',
                                  line=dict(dash='solid')))
        
        # Thêm dữ liệu dự đoán
        fig.add_trace(go.Scatter(x=predictions_restored.index[:n_future],  # Sử dụng n_future để lấy đúng số ngày
                                  y=predictions_restored[feature][:n_future],  # Sử dụng n_future để lấy đúng số ngày
                                  name=f'Predicted {feature}',
                                  line=dict(dash='dash')))
else:
    # Thêm dữ liệu thực tế cho đặc trưng đã chọn
    fig.add_trace(go.Scatter(x=st.session_state['filtered_data'].index[-n_future:],  # Sử dụng n_future để lấy đúng số ngày
                              y=st.session_state['filtered_data'][selected_feature][-n_future:],  # Sử dụng n_future để lấy đúng số ngày
                              name=f'Actual {selected_feature}',
                              line=dict(color='blue')))
    
    # Thêm dữ liệu dự đoán cho đặc trưng đã chọn
    fig.add_trace(go.Scatter(x=predictions_restored.index[:n_future],  # Sử dụng n_future để lấy đúng số ngày
                              y=predictions_restored[selected_feature][:n_future],  # Sử dụng n_future để lấy đúng số ngày
                              name=f'Predicted {selected_feature}',
                              line=dict(color='red', dash='dash')))

fig.update_layout(title=f'{selected_feature} thực tế và dự đoán {selected_feature} các ngày tiếp theo',
                 xaxis_title='Ngày',
                 yaxis_title='Giá',
                 showlegend=True)

st.plotly_chart(fig)

# Hiển thị thống kê
st.subheader("2.3. Thống kê dự đoán")
st.write("Thống kê mô tả:")
st.write(predictions_restored.describe())

# Tính và hiển thị độ biến động
volatility = predictions_restored['Close'].pct_change().std() * np.sqrt(252) * 100
st.metric("Độ biến động hàng năm của giá dự đoán", f"{volatility:.2f}%")

# Xuất dữ liệu
if st.button("Tải xuống dự đoán"):
    predictions_restored.to_csv('predictions_restored.csv')
    st.success("✅ Đã lưu dự đoán vào file 'predictions_restored.csv'")



