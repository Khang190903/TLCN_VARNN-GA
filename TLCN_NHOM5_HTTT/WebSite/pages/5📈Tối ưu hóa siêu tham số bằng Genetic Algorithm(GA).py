import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from deap import base, creator, tools, algorithms
import math
import random
import warnings
import time

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tối ưu hóa siêu tham số bằng GA", page_icon="📈")



# # Thêm nút để xóa kết quả và chạy lại nếu cần
# if st.button("Xóa kết quả và chạy lại"):
#     # Xóa các kết quả từ session state
#     for key in ['optimization_completed', 'best_params', 'best_fitness', 
#                 'best_solutions', 'avg_solutions']:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.rerun()  # Chạy lại trang


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

st.title("Tối ưu hóa siêu tham số bằng Genetic Algorithm")

# Kiểm tra dữ liệu
required_keys = ['train_data', 'valid_data', 'test_data', 'best_lag']
if not all(key in st.session_state for key in required_keys):
    st.warning("⚠️ Vui lòng thực hiện các bước trước đó!")
    st.stop()

# 1. Chuẩn bị dữ liệu
st.header("1. Chuẩn bị dữ liệu cho LSTM")

@st.cache_data
def create_dataset(data, lag_order):
    dataX, dataY = [], []
    for i in range(len(data) - lag_order - 1):
        a = data[i:(i + lag_order), :]
        dataX.append(a)
        dataY.append(data[i + lag_order, :])
    return np.array(dataX), np.array(dataY)

# Lấy dữ liệu từ session state
train_data = st.session_state['train_data']
valid_data = st.session_state['valid_data']
test_data = st.session_state['test_data']
lag_order = st.session_state['best_lag']

# Tạo dataset
with st.spinner("Đang chuẩn bị dữ liệu..."):
    X_train, y_train = create_dataset(train_data.values, lag_order)
    X_valid, y_valid = create_dataset(valid_data.values, lag_order)
    X_test, y_test = create_dataset(test_data.values, lag_order)

    # Reshape dữ liệu
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Hiển thị thông tin về shape của dữ liệu
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Train shape", f"{X_train.shape}")
with col2:
    st.metric("Validation shape", f"{X_valid.shape}")
with col3:
    st.metric("Test shape", f"{X_test.shape}")

# 2. Thiết lập GA
st.header("2. Thiết lập Genetic Algorithm")

# Tham số GA
n_dimensions = 3
POPULATION_SIZE = max(4 + int(3 * np.log(n_dimensions)), 15)
N_GENERATIONS = 7

P_CROSSOVER = 0.8
P_MUTATION = 1.0 / n_dimensions

# Hiển thị tham số GA
col1, col2 = st.columns(2)
with col1:
    st.write("**Tham số cơ bản:**")
    st.write(f"- Số chiều: {n_dimensions}")
    st.write(f"- Kích thước quần thể: {POPULATION_SIZE}")
    st.write(f"- Số thế hệ: {N_GENERATIONS}")
with col2:
    st.write("**Xác suất:**")
    st.write(f"- Xác suất lai ghép: {P_CROSSOVER:.2f}")
    st.write(f"- Xác suất đột biến: {P_MUTATION:.2f}")

# 3. Thực hiện tối ưu hóa
st.header("3. Tối ưu hóa siêu tham số")

def evaluate(individual):
    epochs = int(individual[0])
    batch_size = int(individual[1])
    n_neurons = int(individual[2])

    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_valid, y_valid))
    
    test_predict = model.predict(X_test)
    mse = mean_squared_error(y_test, test_predict)
    rmse = math.sqrt(mse)
    
    return (rmse,)

# Khởi tạo biến dừng tiến trình nếu chưa có
if 'stop_optimization' not in st.session_state:
    st.session_state.stop_optimization = False

# Nút bắt đầu tối ưu hóa
if st.button("Bắt đầu tối ưu hóa"):
    # Khởi tạo biến dừng tiến trình
    st.session_state.stop_optimization = False  # Đặt lại thành False khi bắt đầu tối ưu hóa

    # Nút dừng tiến trình
    if st.button("Dừng tiến trình"):
        st.session_state.stop_optimization = True

    with st.spinner("Đang thực hiện tối ưu hóa..."):
        start_time_optimization = time.time()  # Bắt đầu thời gian tối ưu hóa
        
        # Thiết lập GA
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Định nghĩa các hàm
        toolbox.register("epochs", np.random.randint, 10, 101)
        toolbox.register("batch_size", np.random.randint, 16, 65)
        toolbox.register("n_neurons", np.random.randint, 32, 201)
        
        # Khởi tạo cá thể và quần thể
        def init_individual():
            return [toolbox.epochs(), toolbox.batch_size(), toolbox.n_neurons()]
        
        toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Đăng ký các operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        
        # Khởi tạo quần thể
        population = toolbox.population(n=POPULATION_SIZE)
        best_solutions = []
        avg_solutions = []
        
        # Chạy GA
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        ngens = 7  # Số thế hệ
        for gen in range(ngens): 
            # Cập nhật tiến trình
            progress = (gen + 1) / ngens
            progress_bar.progress(progress)
            status_text.text(f"Đang xử lý thế hệ {gen + 1}/{ngens}")

            # Chọn lọc
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Lai ghép chéo
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < P_CROSSOVER:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    # Đảm bảo giá trị trong khoảng hợp lý
                    child1[:] = [max(10, min(200, int(x))) for x in child1]
                    child2[:] = [max(10, min(200, int(x))) for x in child2]
            
            # Đột biến
            for mutant in offspring:
                if np.random.random() < P_MUTATION:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    mutant[:] = [max(10, min(200, int(x))) for x in mutant]
            
            # Đánh giá fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Cập nhật quần thể
            population[:] = offspring
            
            # Lưu thống kê
            fits = [ind.fitness.values[0] for ind in population]
            best_solutions.append(min(fits))
            avg_solutions.append(sum(fits) / len(fits))

        # Lấy cá thể tốt nhất
        fits = [ind.fitness.values[0] for ind in population]
        best_idx = np.argmin(fits)
        best_individual = population[best_idx]
        best_fitness = fits[best_idx]
        
        # Lưu tất cả kết quả vào session state
        st.session_state['optimization_completed'] = True
        st.session_state['best_params'] = {
            'epochs': int(best_individual[0]),
            'batch_size': int(best_individual[1]),
            'n_neurons': int(best_individual[2])
        }

        # Lưu dữ liệu train/valid/test
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.session_state['X_valid'] = X_valid
        st.session_state['y_valid'] = y_valid
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
    
        st.session_state['best_fitness'] = float(best_fitness)
        st.session_state['best_solutions'] = best_solutions
        st.session_state['avg_solutions'] = avg_solutions
        
        end_time_optimization = time.time()  # Kết thúc thời gian tối ưu hóa
        st.session_state['optimization_time'] = end_time_optimization - start_time_optimization  # Lưu vào session_state


# Thêm nút để hỏi người dùng có muốn xem kết quả đã chạy từ trước không
# if st.session_state.stop_optimization:
if st.button("Xem kết quả đã chạy sẵn trước đó"):
    # Kiểm tra xem file đã tải lên có khớp với tập dữ liệu đã chọn không
    if 'uploaded_file' in st.session_state:
        uploaded_file_name = st.session_state['uploaded_file'].name
        
    # Hiển thị hình ảnh tương ứng với file đã tải lên
        if uploaded_file_name == 'BTC-USD.csv':
            st.image("images/btc_toiuu.png", caption="Dữ liệu BTC-USD")
        elif uploaded_file_name == 'ETH-USD.csv':
            st.image("images/eth_toiuu.png", caption="Dữ liệu ETH-USD")
        elif uploaded_file_name == 'LTC-USD.csv':
            st.image("images/ltc_toiuu.png", caption="Dữ liệu LTC-USD")
        elif uploaded_file_name == 'XRP-USD.csv':
            st.image("images/xrp_toiuu.png", caption="Dữ liệu XRP-USD")
        else:
            st.warning("⚠️ Dữ liệu không xác định!")
    else:
        st.warning("⚠️ Chưa có file nào được tải lên.")

# Hiển thị kết quả nếu đã hoàn thành tối ưu hóa
if 'optimization_completed' in st.session_state and st.session_state['optimization_completed']:
    st.success("✅ Đã hoàn thành tối ưu hóa!")
    st.metric("Thời gian tối ưu hóa", f"{st.session_state['optimization_time']:.2f} giây")
    
    # Hiển thị tham số tối ưu
    st.write("**Tham số tối ưu:**")
    st.write(f"- Epochs: {st.session_state['best_params']['epochs']}")
    st.write(f"- Batch size: {st.session_state['best_params']['batch_size']}")
    st.write(f"- Số neurons: {st.session_state['best_params']['n_neurons']}")
    st.write(f"- RMSE tốt nhất: {st.session_state['best_fitness']:.4f}")

    # Vẽ biểu đồ tiến hóa
    fig, ax = plt.subplots(figsize=(10, 6))
    generations = range(1, len(st.session_state['best_solutions']) + 1)
    ax.plot(generations, st.session_state['best_solutions'], 'b-', label='Best Fitness')
    ax.plot(generations, st.session_state['avg_solutions'], 'r--', label='Average Fitness')
    ax.set_xlabel('Thế hệ')
    ax.set_ylabel('Fitness (RMSE)')
    ax.set_title('Tiến trình tối ưu hóa GA')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Thêm nút để xóa kết quả và chạy lại nếu cần
    if st.button("Xóa kết quả và chạy lại"):
        # Xóa các kết quả từ session state
        for key in ['optimization_completed', 'best_params', 'best_fitness', 
                    'best_solutions', 'avg_solutions']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()  # Chạy lại trang


