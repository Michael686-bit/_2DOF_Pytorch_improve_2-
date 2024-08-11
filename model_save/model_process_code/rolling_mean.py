import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 读取 Excel 文件
df = pd.read_excel('model_save\params_20240810_124251.xlsx')

# 显示 DataFrame 的前几行
print(df.head())

# 访问具体的列
reward_all = df['reward_all'].tolist()

# 计算滑动平均
window_size = 100  # 滑动窗口大小，根据需要调整
rolling_mean = pd.Series(reward_all).rolling(window=window_size).mean()

# 画图
fig1 = plt.figure()  # 创建第一个图形对象
ax1 = fig1.add_subplot(111)  # 在第一个图形对象中添加子图

# 绘制原始数据曲线
ax1.plot(np.arange(len(reward_all)), reward_all, label='reward', color='lightblue', linewidth=2)

# 绘制滑动平均曲线
ax1.plot(np.arange(len(rolling_mean)), rolling_mean, label=f'window_rolling', color='red', linestyle='--')

# 在图形中添加一个矩形
# rect = patches.Rectangle((-42.5-10, 39.23-10), 20, 20, linewidth=2, edgecolor='r', facecolor='none')  # 矩形左下角坐标为(1, 5)，宽度为2，高度为10
# ax1.add_patch(rect)  # 将矩形添加到子图中

ax1.set_xlabel('Step')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.set_title('reward with Moving Average')

plt.show()
