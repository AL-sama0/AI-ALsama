import numpy as np
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# 设置页面标题
st.set_page_config(page_title="房价预测系统", layout="wide")

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# 加载数据
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# 数据探索部分
st.title("California Housing Price Prediction Analysis")

# 显示数据
st.subheader("数据概览")
st.dataframe(data.head())

# 数据分布可视化
st.subheader("价格分布")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['PRICE'], kde=True, bins=20, ax=ax)
ax.set_title("Price Distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 特征相关性热力图
st.subheader("特征相关性热力图")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

# 数据预处理
features = data.drop('PRICE', axis=1)
target = data['PRICE']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,
    target,
    test_size=0.2,
    random_state=42
)

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

st.subheader("线性回归模型性能")
st.write(f"均方误差(MSE): {mse:.4f}")
st.write(f"R²分数: {r2:.4f}")

# 神经网络模型
nn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 训练模型（简化训练过程用于演示）
history = nn_model.fit(
    X_train, y_train,
    epochs=100,  # 减少epochs以便快速演示
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop],
)

# 模型预测
y_pred_nn = nn_model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

st.subheader("神经网络模型性能")
st.write(f"均方误差(MSE): {mse_nn:.4f}")
st.write(f"R²分数: {r2_nn:.4f}")

# 训练过程可视化
st.subheader("训练过程")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 预测界面
st.subheader("房价预测")

# 创建输入控件
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("MedInc (中位收入)", value=3.0)
    house_age = st.number_input("HouseAge (房屋年龄)", value=28.0)
    ave_rooms = st.number_input("AveRooms (平均房间数)", value=5.0)
    ave_bedrms = st.number_input("AveBedrms (平均卧室数)", value=1.0)

with col2:
    population = st.number_input("Population (人口)", value=1500.0)
    ave_occup = st.number_input("AveOccup (平均入住率)", value=3.0)
    latitude = st.number_input("Latitude (纬度)", value=34.0)
    longitude = st.number_input("Longitude (经度)", value=-118.0)

# 预测按钮
if st.button('预测房价'):
    # 准备输入数据
    input_features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                                population, ave_occup, latitude, longitude]])

    # 标准化特征
    input_scaled = scaler.transform(input_features)

    # 使用神经网络模型预测
    prediction = nn_model.predict(input_scaled)

    # 显示结果
    st.success(f"预测房价: ${prediction[0][0] * 100000:.2f}")

    # 同时显示线性回归结果对比
    lr_prediction = lr_model.predict(input_scaled)
    st.info(f"线性回归预测: ${lr_prediction[0] * 100000:.2f}")

# 保存模型
nn_model.save('housing_model.h5')
st.success("模型已保存为 'housing_model.h5'")