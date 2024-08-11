# data_process\excel_test\data.xlsx

import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('data_process\excel_test\data.xlsx')

# 显示 DataFrame 的前几行
print(df.head())

# 访问具体的列
x_vals = df['x_vals'].tolist()
y_vals = df['y_vals'].tolist()

print(f"x_vals = {x_vals}")
print(f"y_vals = {y_vals}")


