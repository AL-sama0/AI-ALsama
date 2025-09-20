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


# ==================== 数据加载和探索 ====================
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

print("数据前5行：")
print(data.head())
print("\n" + "="*50)


plt.figure(figsize=(10, 6))
sns.histplot(data['PRICE'], kde=True, bins=20)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


print("缺失值检查：")
print(data.isnull().sum())
print("\n" + "="*50)


# ==================== 数据预处理 ====================
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


# ==================== 线性回归模型 ====================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("线性回归模型性能评估：")
print(f"均方误差(MSE): {mse}")
print(f"R2分数: {r2}")
print("\n" + "="*50)


# ==================== 神经网络模型 ====================
nn_model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 关键修改：降低学习率，使用更小的学习率
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("神经网络模型结构：")
nn_model.summary()
print("\n" + "="*50)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 关键修改：增加批量大小，减少训练轮数
history = nn_model.fit(
    X_train, y_train,
    epochs=500,  # 增加总轮数，让早停来决定何时停止
    batch_size=256,  # 大幅增加批量大小，使训练更稳定
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop],
)


# ==================== 神经网络预测和评估 ====================
predictions = nn_model.predict(X_test)

print("前5个预测结果对比：")
for i in range(5):
    print(f"预测值: {predictions[i][0]:.2f}, 实际值: {y_test.iloc[i]:.2f}, 误差: {(y_test.iloc[i] - predictions[i][0]):.2f}")

# 绘制更光滑的曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, alpha=0.8)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Changes During Training (Smooth Curve)')
plt.grid(True, alpha=0.3)
plt.show()