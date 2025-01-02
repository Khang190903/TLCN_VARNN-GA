import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_var_metrics_detailed(train_data, valid_data, lag):
   model_var = VAR(train_data)
   var_result = model_var.fit(lag)
   
   # Khởi tạo mảng để lưu dự đoán
   forecasts = []
   last_values = train_data.values[-lag:]  # Lấy lag giá trị cuối cùng của tập train
    # Dự đoán rolling
   for i in range(len(valid_data)):
       # Dự đoán giá trị tiếp theo
       forecast = var_result.forecast(last_values, steps=1)[0]
       forecasts.append(forecast)
       
       # Cập nhật last_values với giá trị thực tế từ valid_data
       actual_value = valid_data.iloc[i].values.reshape(1, -1)
       last_values = np.vstack([last_values[1:], actual_value])
    # Chuyển dự đoán thành DataFrame
   forecast_df = pd.DataFrame(forecasts, index=valid_data.index, columns=valid_data.columns)
   
   # Tính các metrics cho từng cột
   metrics = {}
   for column in valid_data.columns:
       rmse = np.sqrt(mean_squared_error(valid_data[column], forecast_df[column]))
       mae = mean_absolute_error(valid_data[column], forecast_df[column])
       mape = np.mean(np.abs((valid_data[column] - forecast_df[column]) / valid_data[column])) * 100
       cv_rmse = rmse / np.mean(valid_data[column]) * 100
    #    metrics[column] = {'RMSE': rmse, 'MAE': mae, 'CV(RMSE)': cv_rmse}
       metrics[column] = {'RMSE': rmse, 'MAE': mae}

   
   return metrics, forecast_df

def display_lag_analysis(tab, metrics, forecast_df, lag):
    with tab:
        # Tạo layout 2 cột cho metrics
        col1, col2 = st.columns(2)
        
        # Hiển thị metrics trong khung màu với định dạng đẹp hơn
        with col1:
            st.markdown("""
                <style>
                .metric-box {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                }
                </style>
                """, unsafe_allow_html=True)
            
            for column in list(metrics.keys())[:len(metrics)//2 + len(metrics)%2]:
                st.markdown(f"""
                    <div class="metric-box">
                        <h3>{column}</h3>
                        <p>RMSE: {metrics[column]['RMSE']:.4f}</p>
                        <p>MAE: {metrics[column]['MAE']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            for column in list(metrics.keys())[len(metrics)//2 + len(metrics)%2:]:
                st.markdown(f"""
                    <div class="metric-box">
                        <h3>{column}</h3>
                        <p>RMSE: {metrics[column]['RMSE']:.4f}</p>
                        <p>MAE: {metrics[column]['MAE']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
# Thêm selectbox với tùy chọn "Xem tất cả"
        options = list(metrics.keys()) + ["ALL"]
        selected_variable = st.selectbox(
            "Chọn biến để hiển thị biểu đồ",
            options=options,
            key=f"select_var_{lag}"  
        )
        
        st.subheader("Biểu đồ so sánh giá trị dự đoán và giá trị thực tế cho các độ trễ")

        if selected_variable == "ALL":
            # Tạo subplot cho tất cả các biến
            n_vars = len(metrics.keys())
            fig, axes = plt.subplots(n_vars, 1, figsize=(12, 5*n_vars))
            
            for idx, var in enumerate(metrics.keys()):
                valid_values = st.session_state['valid_data'][var]
                
                axes[idx].plot(valid_values.index, valid_values, 
                        label='Thực tế', color='#2E86C1', linewidth=2)
                axes[idx].plot(forecast_df.index, forecast_df[var], 
                        label='Dự đoán', color='#E74C3C', linestyle='--', linewidth=2)
                
                axes[idx].set_title(f'{var} - Độ trễ {lag}', pad=20, fontsize=12, fontweight='bold')
                axes[idx].legend(loc='upper right', fontsize=10)
                axes[idx].grid(True, linestyle='--', alpha=0.7)
                axes[idx].set_xlabel('Thời gian', fontsize=10)
                axes[idx].set_ylabel('Giá trị', fontsize=10)
                axes[idx].tick_params(axis='x', rotation=45)
            
        else:
            # Hiển thị biểu đồ cho biến được chọn
            fig, ax = plt.subplots(figsize=(12, 5))
            
            valid_values = st.session_state['valid_data'][selected_variable]
            
            ax.plot(valid_values.index, valid_values, 
                    label='Thực tế', color='#2E86C1', linewidth=2)
            ax.plot(forecast_df.index, forecast_df[selected_variable], 
                    label='Dự đoán', color='#E74C3C', linestyle='--', linewidth=2)
            
            ax.set_title(f'{selected_variable} - Độ trễ {lag}', pad=20, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Thời gian', fontsize=10)
            ax.set_ylabel('Giá trị', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)


st.set_page_config(
    page_title="Xác định độ trễ (LAG)",
    page_icon="🎈",
)

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

st.title("Xác định độ trễ tối ưu (LAG)")

# Kiểm tra dữ liệu
if not all(key in st.session_state for key in ['train_data', 'valid_data', 'df_log_diff']):
    st.warning("⚠️ Vui lòng thực hiện các bước trước đó!")
    st.stop()

# Lấy dữ liệu từ session state
train_data = st.session_state['train_data']
valid_data = st.session_state['valid_data']
df_log_diff = st.session_state['df_log_diff']

# 1. Phân tích độ trễ tối ưu
st.header("1. Phân tích độ trễ tối ưu")
max_lags = st.slider("Số độ trễ tối đa để kiểm tra", 5, 20, 20)

with st.spinner("Đang phân tích độ trễ tối ưu..."):
    model = VAR(train_data)
    # Tạo list để lưu các giá trị cho từng độ trễ
    aic_values = []
    bic_values = []
    fpe_values = []
    hqic_values = []
    
    # Tính các chỉ số cho từng độ trễ
    for lag in range(1, max_lags + 1):
        result = model.fit(lag)
        aic_values.append(result.aic)
        bic_values.append(result.bic)
        fpe_values.append(result.fpe)
        hqic_values.append(result.hqic)
    
    # Tạo DataFrame để hiển thị kết quả
    lag_metrics = pd.DataFrame({
        'Độ trễ': range(1, max_lags + 1),
        'AIC': aic_values,
        'BIC': bic_values,
        'FPE': fpe_values,
        'HQIC': hqic_values
    }).set_index('Độ trễ')
    
    # # Hiển thị bảng kết quả
    # st.dataframe(lag_metrics.round(4))
    
    # Tìm độ trễ tối ưu theo từng tiêu chí
    optimal_lags = {
        'AIC': lag_metrics['AIC'].idxmin(),
        'BIC': lag_metrics['BIC'].idxmin(),
        'FPE': lag_metrics['FPE'].idxmin(),
        'HQIC': lag_metrics['HQIC'].idxmin()
    }
    
    # Hiển thị độ trễ tối ưu theo từng tiêu chí
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Độ trễ tối ưu theo AIC", f"{optimal_lags['AIC']}")
        st.metric("Độ trễ tối ưu theo BIC", f"{optimal_lags['BIC']}")
    with col2:
        st.metric("Độ trễ tối ưu theo FPE", f"{optimal_lags['FPE']}")
        st.metric("Độ trễ tối ưu theo HQIC", f"{optimal_lags['HQIC']}")

# 2. So sánh hiệu suất với các độ trễ khác nhau
st.header("2. So sánh hiệu suất các độ trễ")

# Cho phép người dùng chọn các độ trễ để so sánh
selected_lags = st.multiselect(
    "Chọn các độ trễ để so sánh",
    options=list(set(optimal_lags.values())),  # Chỉ hiện các độ trễ tối ưu
    default=list(set(optimal_lags.values()))   # Mặc định chọn tất cả các độ trễ tối ưu
)

if not selected_lags:
    st.warning("Vui lòng chọn ít nhất một độ trễ để phân tích")
    st.stop()

# Tạo tabs động dựa trên độ trễ được chọn
tabs = st.tabs([f"Độ trễ {lag}" for lag in selected_lags])

# Tính toán metrics cho mỗi độ trễ được chọn
metrics_dict = {}
forecast_dict = {}
for lag in selected_lags:
    metrics_dict[lag], forecast_dict[lag] = calculate_var_metrics_detailed(train_data, valid_data, lag)

# Hiển thị phân tích trong từng tab
for tab, lag in zip(tabs, selected_lags):
    display_lag_analysis(tab, metrics_dict[lag], forecast_dict[lag], lag)


# 3. Kết luận về độ trễ tối ưu
st.header("3. Kết luận")

# Tính trung bình RMSE cho mỗi độ trễ được chọn
avg_rmse = {
    lag: np.mean([metrics_dict[lag][col]['RMSE'] for col in metrics_dict[lag]])
    for lag in selected_lags
}

# Tìm độ trễ tối ưu dựa trên RMSE
best_lag = min(avg_rmse, key=avg_rmse.get)
best_rmse = avg_rmse[best_lag]

# Hiển thị bảng so sánh chi tiết
st.subheader("So sánh chi tiết các độ trễ")
comparison_df = pd.DataFrame({
    'Độ trễ': selected_lags,
    'RMSE trung bình': [avg_rmse[lag] for lag in selected_lags]
})

# Thêm RMSE cho từng cột dữ liệu
for col in train_data.columns:
    comparison_df[f'{col} RMSE'] = [metrics_dict[lag][col]['RMSE'] for lag in selected_lags]

st.dataframe(comparison_df.round(4))

# Vẽ biểu đồ so sánh RMSE trung bình cho các độ trễ theo kiểu sin
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(comparison_df['Độ trễ'].astype(str), comparison_df['RMSE trung bình'], marker='o', color='#2E86C1', linestyle='-')
ax.set_title('So sánh RMSE trung bình theo độ trễ', fontsize=16)
ax.set_xlabel('Độ trễ', fontsize=12)
ax.set_ylabel('RMSE trung bình', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

# Hiển thị biểu đồ
st.pyplot(fig)

# Hiển thị kết luận
st.success(f"🎉 Độ trễ tối ưu là {best_lag} với RMSE trung bình = {best_rmse:.4f}")

# Lưu kết quả vào session state
st.session_state['best_lag'] = best_lag
st.session_state['best_rmse'] = best_rmse



