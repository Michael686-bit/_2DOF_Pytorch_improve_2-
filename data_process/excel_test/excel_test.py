import pandas as pd

# 示例数据
x_vals = [1, 2, 3, 4, 5,1001]
y_vals = [2, 3, 5, 7, 11,1001]

# 创建 DataFrame
df = pd.DataFrame({
    'x_vals': x_vals,
    'y_vals': y_vals
})

# 保存 DataFrame 到 Excel 文件
df.to_excel('data_process\data\datanew.xlsx', index=False)
