

import time  # Nhập mô-đun time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inspect import signature
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_page_config(page_title="Huấn luyện mô hình VARNN", page_icon="🤖")

# # Thêm nút reset nếu muốn huấn luyện lại
# if st.button("Reset kết quả huấn luyện"):
#     for key in ['model_optimized_trained', 'model_default_trained', 
#                 'model_optimized', 'model_default',
#                 'history_optimized', 'history_default',
#                 'metrics_optimized', 'metrics_default']:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.rerun()


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

st.title("Huấn luyện mô hình VARNN")

# Kiểm tra dữ liệu
required_keys = ['X_train', 'y_train', 'X_valid', 'y_valid', 'X_test', 'y_test', 'best_params']
if not all(key in st.session_state for key in required_keys):
    st.warning("⚠️ Vui lòng thực hiện các bước trước đó!")
    st.stop()

if 'model_default' not in st.session_state:
    st.session_state['model_default'] = None  # Hoặc khởi tạo với giá trị mặc định khác nếu cần
# 1. Huấn luyện mô hình với tham số tối ưu từ GA
st.header("1. Mô hình VARNN với tham số tối ưu từ GA")
# Hiển thị tham số tối ưu
st.subheader("1.1. Tham số tối ưu từ GA")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Epochs", st.session_state['best_params']['epochs'])
with col2:
    st.metric("Batch Size", st.session_state['best_params']['batch_size'])
with col3:
    st.metric("Neurons", st.session_state['best_params']['n_neurons'])
# Kết hợp tập train và validation
X_final_train = np.concatenate((st.session_state['X_train'], st.session_state['X_valid']), axis=0)
y_final_train = np.concatenate((st.session_state['y_train'], st.session_state['y_valid']), axis=0)
# Kiểm tra xem đã huấn luyện mô hình tối ưu chưa
if 'model_optimized_trained' not in st.session_state:
    st.session_state['model_optimized_trained'] = False
if st.button("Huấn luyện mô hình VARNN với tham số tối ưu") or st.session_state['model_optimized_trained']:
    if not st.session_state['model_optimized_trained']:
        with st.spinner("Đang huấn luyện mô hình..."):
            start_time = time.time()  # Thời gian bắt đầu huấn luyện
            # Xây dựng và huấn luyện mô hình LSTM tối ưu
            model_optimized = Sequential()
            model_optimized.add(LSTM(st.session_state['best_params']['n_neurons'], 
                                    input_shape=(X_final_train.shape[1], X_final_train.shape[2]), 
                                    activation='relu'))
            model_optimized.add(Dense(4))
            model_optimized.compile(loss='mean_squared_error', optimizer='adam')
                # Huấn luyện mô hình
            history_optimized = model_optimized.fit(
                X_final_train, y_final_train,
                epochs=st.session_state['best_params']['epochs'],
                batch_size=st.session_state['best_params']['batch_size'],
                verbose=0
            )

            end_time = time.time()  # Thời gian kết thúc huấn luyện
            training_time = end_time - start_time  # Thời gian huấn luyện

            # Lưu thời gian vào session state
            st.session_state['training_time'] = training_time

            # Dự đoán trên tập train
            predictions_train = model_optimized.predict(X_final_train)
            
            # Tính các metrics trên tập train
            mse_train = mean_squared_error(y_final_train, predictions_train)
            rmse_train = math.sqrt(mse_train)
            mae_train = mean_absolute_error(y_final_train, predictions_train)
            cv_rmse_train = (rmse_train / np.mean(y_final_train)) * 100
            # Dự đoán trên tập test
            predictions_optimized = model_optimized.predict(st.session_state['X_test'])
            
            # Chuyển đổi ngược về dạng gốc
            y_test_original = np.exp(np.cumsum(st.session_state['y_test'], axis=0)) - 1
            predictions_original = np.exp(np.cumsum(predictions_optimized, axis=0)) - 1
            
            # Tính các metrics trên dữ liệu gốc
            mse_optimized = mean_squared_error(y_test_original, predictions_original)
            rmse_optimized = math.sqrt(mse_optimized)
            mae_optimized = mean_absolute_error(y_test_original, predictions_original)
            cv_rmse_optimized = (rmse_optimized / np.mean(y_test_original)) * 100
            
            # In ra các giá trị để kiểm tra
            st.write(f"Giá trị trung bình của y_test_original: {np.mean(y_test_original)}")
            st.write(f"RMSE: {rmse_optimized}")
            
            
            # Đảm bảo trung bình của y_test_original không bằng 0 để tránh chia cho 0
            mean_y_test_original = np.mean(y_test_original)
            if mean_y_test_original != 0:
                cv_rmse_optimized = (rmse_optimized / mean_y_test_original) * 100
            else:
                cv_rmse_optimized = float('inf')  # Gán vô cực nếu trung bình bằng 0

            # Lưu kết quả vào session state
            st.session_state['metrics_optimized'] = {
                'mse': mse_optimized,  # Thêm MSE vào session state
                'rmse': rmse_optimized,
                'mae': mae_optimized,
                'cv_rmse': cv_rmse_optimized
            }

            st.session_state['model_optimized'] = model_optimized
            st.session_state['history_optimized'] = history_optimized.history
            st.session_state['model_optimized_trained'] = True

            # Hiển thị thời gian huấn luyện  trên tập train
            st.success(f"Thời gian huấn luyện: {training_time:.4f} giây")
            st.metric("Train MSE", f"{mse_train:.6f}")
            st.metric("Train RMSE", f"{rmse_train:.6f}")
            st.metric("Train MAE", f"{mae_train:.6f}")
            st.metric("Train CV(RMSE) %", f"{cv_rmse_train:.2f}")

            # # Hiển thị metrics trên tập test
            # st.metric("Test MSE", f"{mse_optimized:.6f}")
            # st.metric("Test RMSE", f"{rmse_optimized:.6f}")
            # st.metric("Test MAE", f"{mae_optimized:.6f}")
            # st.metric("Test CV(RMSE) %", f"{cv_rmse_optimized:.2f}")
# # Khởi tạo các khóa trong session_state nếu chưa tồn tại
# if 'training_time' not in st.session_state:
#     st.session_state['training_time'] = 0.0
        # Hiển thị metrics
if 'metrics_optimized' in st.session_state:
    st.subheader("1.2. Kết quả đánh giá mô hình tối ưu")
    col1, col2, col3, col4 = st.columns(4)  # Thêm cột cho MSE
    with col1:
        st.metric("MSE", f"{st.session_state['metrics_optimized']['mse']:.6f}")  # Hiển thị MSE
    with col2:
        st.metric("RMSE", f"{st.session_state['metrics_optimized']['rmse']:.6f}")
    with col3:
        st.metric("MAE", f"{st.session_state['metrics_optimized']['mae']:.6f}")
    with col4:
        st.metric("CV(RMSE) %", f"{st.session_state['metrics_optimized']['cv_rmse']:.2f}")

    st.metric("Thời gian huấn luyện", f"{st.session_state.get('training_time', 0):.4f} giây") 

        # Vẽ biểu đồ loss
    st.subheader("1.3. Biểu đồ Loss trong quá trình huấn luyện")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state['history_optimized']['loss'])
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    st.pyplot(fig)
# 2. Huấn luyện mô hình với tham số mặc định
st.header("2. Mô hình VARNN với tham số mặc định")


#Hiển thị tham số mặc định
st.subheader("2.1. Tham số mặc định")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Epochs", 200)
with col2:
    st.metric("Batch Size", 32)
with col3:
    st.metric("Neurons", 50)
# Kiểm tra xem đã huấn luyện mô hình mặc định chưa
if 'model_default_trained' not in st.session_state:
    st.session_state['model_default_trained'] = False
if st.button("Huấn luyện mô hình VARNN với tham số mặc định") or st.session_state['model_default_trained']:
    if not st.session_state['model_default_trained']:
        with st.spinner("Đang huấn luyện mô hình mặc định..."):
            start_time = time.time()  # Thời gian bắt đầu huấn luyện
            # Xây dựng mô hình LSTM mặc định
            model_default = Sequential()
            model_default.add(LSTM(50, input_shape=(X_final_train.shape[1], X_final_train.shape[2]), 
                                activation='relu'))
            model_default.add(Dense(4))
            model_default.compile(loss='mean_squared_error', optimizer='adam')
                # Huấn luyện mô hình
            history_default = model_default.fit(
                X_final_train, y_final_train,
                epochs=200,
                batch_size=32,
                verbose=0
            )

            end_time = time.time()  # Thời gian kết thúc huấn luyện
            training_time_default = end_time - start_time  # Thời gian huấn luyện

            # Lưu thời gian vào session state
            st.session_state['training_time_default'] = training_time_default

            # Dự đoán trên tập train
            predictions_train = model_default.predict(X_final_train)
            
            # Tính các metrics trên tập train
            mse_train = mean_squared_error(y_final_train, predictions_train)
            rmse_train = math.sqrt(mse_train)
            mae_train = mean_absolute_error(y_final_train, predictions_train)
            cv_rmse_train = (rmse_train / np.mean(y_final_train)) * 100


            # Dự đoán trên tập test
            predictions_default = model_default.predict(st.session_state['X_test'])
            
            # Chuyển đổi ngược về dạng gốc
            y_test_original = np.exp(np.cumsum(st.session_state['y_test'], axis=0)) - 1
            predictions_original = np.exp(np.cumsum(predictions_default, axis=0)) - 1
            
            # Tính các metrics trên dữ liệu gốc
            mse_default = mean_squared_error(y_test_original, predictions_original)
            rmse_default = math.sqrt(mse_default)
            mae_default = mean_absolute_error(y_test_original, predictions_original)
            cv_rmse_default = (rmse_default / np.mean(y_test_original)) * 100

            # In ra các giá trị để kiểm tra
            st.write(f"Giá trị trung bình của y_test_original: {np.mean(y_test_original)}")
            st.write(f"RMSE: {rmse_default}")
            
            # Đảm bảo trung bình của y_test_original không bằng 0 để tránh chia cho 0
            mean_y_test_original = np.mean(y_test_original)
            if mean_y_test_original != 0:
                cv_rmse_default = (rmse_default / mean_y_test_original) * 100
            else:
                cv_rmse_default = float('inf')  # Gán vô cực nếu trung bình bằng 0

            # Lưu kết quả vào session state
            st.session_state['metrics_default'] = {
                'mse': mse_default,  # Thêm MSE vào session state
                'rmse': rmse_default,
                'mae': mae_default,
                'cv_rmse': cv_rmse_default
            }

            st.session_state['model_default'] = model_default
            st.session_state['history_default'] = history_default.history  # Lưu history
            st.session_state['model_default_trained'] = True

            # Hiển thị thời gian huấn luyện  trên tập train
            st.success(f"Thời gian huấn luyện: {training_time_default:.4f} giây")
            st.metric("Train MSE", f"{mse_train:.6f}")
            st.metric("Train RMSE", f"{rmse_train:.6f}")
            st.metric("Train MAE", f"{mae_train:.6f}")
            st.metric("Train CV(RMSE) %", f"{cv_rmse_train:.2f}")

            # # Hiển thị thời gian huấn luyện và metrics
            # st.success(f"Thời gian huấn luyện: {training_time:.4f} giây")
            # st.metric("MSE", f"{mse_default:.6f}")
            # st.metric("RMSE", f"{rmse_default:.6f}")
            # st.metric("MAE", f"{mae_default:.6f}")
            # st.metric("CV(RMSE) %", f"{cv_rmse_default:.2f}")

# if 'training_time_default' not in st.session_state:
#     st.session_state['training_time_default'] = 0.0

if 'metrics_default' in st.session_state:
    # Hiển thị metrics
    st.subheader("2.2. Kết quả đánh giá mô hình mặc định")
    col1, col2, col3, col4 = st.columns(4)  # Thêm cột cho MSE
    with col1:
        st.metric("MSE", f"{st.session_state['metrics_default']['mse']:.6f}")  # Hiển thị MSE
    with col2:
        st.metric("RMSE", f"{st.session_state['metrics_default']['rmse']:.6f}")
    with col3:
        st.metric("MAE", f"{st.session_state['metrics_default']['mae']:.6f}")
    with col4:
        st.metric("CV(RMSE) %", f"{st.session_state['metrics_default']['cv_rmse']:.2f}")

    st.metric("Thời gian huấn luyện", f"{st.session_state['training_time_default']:.4f} giây")
    # Vẽ biểu đồ loss
    st.subheader("2.2. Biểu đồ Loss trong quá trình huấn luyện")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state['history_default']['loss'])
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    st.pyplot(fig)


# Thêm nút reset nếu muốn huấn luyện lại
if st.button("Reset kết quả huấn luyện"):
    for key in ['model_optimized_trained', 'model_default_trained', 
                'model_optimized', 'model_default',
                'history_optimized', 'history_default',
                'metrics_optimized', 'metrics_default']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# 3. So sánh hai mô hình
if 'metrics_optimized' in st.session_state and 'metrics_default' in st.session_state:
    

    st.header("3. Đánh giá giữa VAR_LSTM_GA và VAR_LSTM")

        # 1. Tính thời gian dự đoán cho model_default (VAR_LSTM)
    start_time_default = time.time()  # Thời gian bắt đầu
    test_predict_default = st.session_state['model_default'].predict(st.session_state['X_test'])
    end_time_default = time.time()  # Thời gian kết thúc

    # Tính thời gian kiểm tra
    test_time_default = end_time_default - start_time_default

    # 2. Tính thời gian dự đoán cho model_optimized (VAR_LSTM_GA)
    start_time_ga = time.time()  # Thời gian bắt đầu
    test_predict_ga = st.session_state['model_optimized'].predict(st.session_state['X_test'])
    end_time_ga = time.time()  # Thời gian kết thúc

    # Tính thời gian kiểm tra
    test_time_ga = end_time_ga - start_time_ga

    # Hiển thị kết quả
    st.subheader("3.1 Thời gian kiểm tra")
    st.write(f"Thời gian kiểm tra cho VAR_LSTM: {test_time_default:.4f} giây")
    st.write(f"Thời gian kiểm tra cho VAR_LSTM_GA: {test_time_ga:.4f} giây")



    st.subheader("3.2 Bảng so sánh chỉ số")
    comparison_data = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'CV(RMSE) %'],
        'VARNN tối ưu': [
            st.session_state['metrics_optimized']['mse'],
            st.session_state['metrics_optimized']['rmse'],
            st.session_state['metrics_optimized']['mae'],
            st.session_state['metrics_optimized']['cv_rmse']
        ],
        'VARNN mặc định': [
            st.session_state['metrics_default']['mse'],
            st.session_state['metrics_default']['rmse'],
            st.session_state['metrics_default']['mae'],
            st.session_state['metrics_default']['cv_rmse']
        ]
    })
    
    st.dataframe(comparison_data.style.format({
        'VARNN tối ưu': '{:.6f}',
        'VARNN mặc định': '{:.6f}'
    }))


# 2. Kết luận so sánh
st.subheader("3.3. Kết luận so sánh")

# Kiểm tra xem metrics_default có tồn tại không trước khi tính toán sự khác biệt
if 'metrics_optimized' in st.session_state and 'metrics_default' in st.session_state:
    # Tính sự khác biệt giữa các metrics
    metrics_diff = {
        'RMSE': st.session_state['metrics_optimized']['rmse'] - st.session_state['metrics_default']['rmse'],
        'MSE': st.session_state['metrics_optimized']['mse'] - st.session_state['metrics_default']['mse'],
        'MAE': st.session_state['metrics_optimized']['mae'] - st.session_state['metrics_default']['mae'],
        'CV(RMSE)': st.session_state['metrics_optimized']['cv_rmse'] - st.session_state['metrics_default']['cv_rmse']  # Thêm CV(RMSE) difference
    }

    # Hiển thị sự khác biệt
    col1, col2, col3, col4 = st.columns(4)  # Thêm cột cho CV(RMSE) Difference
    with col1:
        st.metric("RMSE Difference", f"{metrics_diff['RMSE']:.6f}")
    with col2:
        st.metric("MSE Difference", f"{metrics_diff['MSE']:.6f}")
    with col3:
        st.metric("MAE Difference", f"{metrics_diff['MAE']:.6f}")
    with col4:
        st.metric("CV(RMSE) Difference", f"{metrics_diff['CV(RMSE)']:.6f}")  

    # Đưa ra kết luận
    if all(diff < 0 for diff in metrics_diff.values()):
        st.success("✅ Mô hình VAR kết hợp LSTM với tham số tối ưu từ GA cho kết quả tốt hơn so với VAR kết hợp LSTM mặc định.")
    elif all(diff > 0 for diff in metrics_diff.values()):
        st.error("❌ Mô hình VAR kết hợp LSTM với tham số tối ưu từ GA cho kết quả kém hơn mô hình VAR kết hợp LSTM mặc định.")
    else:
        st.info("ℹ️ Mô hình VAR kết hợp LSTM với tham số tối ưu từ GA có sự cải thiện ở một số chỉ số nhưng không hoàn toàn vượt trội so với mô hình LSTM mặc định.")
else:
    st.warning("⚠️ Vui lòng huấn luyện cả hai mô hình trước khi so sánh.")
