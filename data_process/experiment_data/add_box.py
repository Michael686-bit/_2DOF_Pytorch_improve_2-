# data_process\excel_test\data.xlsx

import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('data_process\experiment_data\Exy_q12_20240810_101304.xlsx')

# 显示 DataFrame 的前几行
print(df.head())

# 访问具体的列
x_vals = (df['x_vals']).tolist()
y_vals = (df['y_vals']).tolist()

# print(f"x_vals = {x_vals}")
# print(f"y_vals = {y_vals}")

# 画图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 第一部分：绘制二维曲线
fig1 = plt.figure()  # 创建第一个图形对象
ax1 = fig1.add_subplot(111)  # 在第一个图形对象中添加子图
ax1.plot(x_vals, y_vals)  #, label='end-effector trajectory'

flag_shape = 0       # 0  矩形    1  圆形
if flag_shape==0:

# 在图形中添加一个矩形 200-42.5, 220+39.23

    rect = patches.Rectangle((-42.5-10, 39.23-10), 20, 20, linewidth=2, edgecolor='r', facecolor='none')  # 矩形左下角坐标为(1, 5)，宽度为2，高度为10
    ax1.add_patch(rect)  # 将矩形添加到子图中
else:

    # 在图形中添加一个圆形
    circle = patches.Circle((-42.5+5, 39.23), radius=10, linewidth=2, edgecolor='r', facecolor='none')  # 圆心坐标为(2, 15)，半径为5
    ax1.add_patch(circle)  # 将圆形添加到子图中

ax1.set_xlabel('X axis/cm')
ax1.set_ylabel('Y axis/cm')
ax1.legend()
ax1.set_title('end-effector trajectory')

# 第二部分：绘制关节角度图像
fig2 = plt.figure()  # 创建第二个图形对象
ax2 = fig2.add_subplot(111)  # 在第二个图形对象中添加子图
q1_vals = df['q1_vals'].tolist()
q2_vals = df['q2_vals'].tolist()
# 对数据进行处理
q1_vals = [0 if x > 6.18 else x for x in q1_vals]
q2_vals = [0 if x > 6.18 else x for x in q2_vals]

ax2.plot(q1_vals, q2_vals, label='Joint Angles')
ax2.set_xlabel('θ1_vals/rad')
ax2.set_ylabel('θ2_vals/rad')
ax2.legend()
ax2.set_title('Joint Angles Plot')

# 显示所有图形
plt.show()