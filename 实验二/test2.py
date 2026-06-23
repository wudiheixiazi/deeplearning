import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 读取数据（使用正确的路径）
file_path = r'C:\Users\22314\Desktop\test_2\train_set.csv'
df_raw = pd.read_csv(file_path)

print(f"原始数据大小: {df_raw.shape}")

# 2. 拆分标签和文本（因为数据只有一列，用制表符分隔）
col_name = df_raw.columns[0]
split_data = df_raw[col_name].str.split('\t', n=1, expand=True)
df = pd.DataFrame()
df['label'] = split_data[0].astype(int)
df['text'] = split_data[1]

print(f"处理后的数据大小: {df.shape}")
print(df.head())

# 3. 划分训练集和测试集 (8:2)
X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持类别分布一致
)

print(f"\n划分结果:")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 4. 保存划分后的数据集（保存到同一目录）
output_dir = r'C:\Users\22314\Desktop\test_2'

train_df = pd.DataFrame({'label': y_train, 'text': X_train})
test_df = pd.DataFrame({'label': y_test, 'text': X_test})

train_df.to_csv(os.path.join(output_dir, 'train_set_8.csv'), index=False, encoding='utf-8')
test_df.to_csv(os.path.join(output_dir, 'test_set_2.csv'), index=False, encoding='utf-8')

print(f"\n已保存到:")
print(f"训练集: {output_dir}\\train_set_8.csv")
print(f"测试集: {output_dir}\\test_set_2.csv")

# 5. 验证保存的数据
print("\n验证保存的数据:")
train_check = pd.read_csv(os.path.join(output_dir, 'train_set_8.csv'))
test_check = pd.read_csv(os.path.join(output_dir, 'test_set_2.csv'))
print(f"训练集大小: {train_check.shape}")
print(f"测试集大小: {test_check.shape}")
print(f"\n训练集标签分布:\n{train_check['label'].value_counts().sort_index()}")
print(f"\n测试集标签分布:\n{test_check['label'].value_counts().sort_index()}")