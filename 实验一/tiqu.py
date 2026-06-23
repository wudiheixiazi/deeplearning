import pandas as pd

# 1. 加载原始完整数据
original = pd.read_csv('./train.csv')  # 原始文件，有所有id和标签

# 2. 加载测试集（只有id，无标签）
test = pd.read_csv('./test_set.csv')    # 只有id列

# 3. 根据测试集的id，从原始数据中提取标签
test_with_labels = test.merge(original[['id', 'label']], on='id', how='left')

# 4. 检查是否全部匹配
print(f"测试集样本数: {len(test)}")
print(f"成功匹配标签数: {test_with_labels['label'].notna().sum()}")
print(f"未匹配数: {test_with_labels['label'].isna().sum()}")

# 5. 保存为 111.csv
id_label_table = test_with_labels[['id', 'label']]
id_label_table.to_csv('./111.csv', index=False)  # ← 改成 111.csv

print("\nid-标签对应表已保存到 111.csv")
print(id_label_table.head())
print(f"\n标签分布:\n{id_label_table['label'].value_counts()}")