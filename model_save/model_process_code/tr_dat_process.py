# data_process\excel_test\data.xlsx

import pandas as pd
import numpy as np
# 读取 Excel 文件
df = pd.read_excel('model_save\params_20240810_105513.xlsx')

# 显示 DataFrame 的前几行
print(df.head())

# 访问具体的列
reward_all = (df['reward_all']).tolist()

# 计算滑动平均
window_size = 10  # 滑动窗口大小，根据需要调整
rolling_mean = pd.Series(reward_all).rolling(window=window_size).mean()

# 画图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 第一部分：绘制二维曲线
fig1 = plt.figure()  # 创建第一个图形对象
ax1 = fig1.add_subplot(111)  # 在第一个图形对象中添加子图
ax1.plot(np.arange(len(reward_all)), reward_all, label='Data Curve')

ax1.set_xlabel('step')
ax1.set_ylabel('reward')
ax1.legend()
ax1.set_title('2D Curve Plot')


# 第二部分：绘制关节角度图像
fig2 = plt.figure()  # 创建第二个图形对象
ax2 = fig2.add_subplot(111)  # 在第二个图形对象中添加子图



ax2.plot(q1_vals, q2_vals, label='Joint Angles')
ax2.set_xlabel('q1_vals')
ax2.set_ylabel('q2_vals')
ax2.legend()
ax2.set_title('Joint Angles Plot')


plt.show()