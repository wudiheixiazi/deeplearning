import pandas as pd
import numpy as np

# ==================== 检查数据 ====================

print("=" * 60)
print("数据检查")
print("=" * 60)

# 1. 检查真实标签
true_df = pd.read_csv('111.csv')
print("\n【真实标签 111.csv】")
print(f"样本数量: {len(true_df)}")
print(f"前10行:")
print(true_df.head(10))
print(f"\n标签分布:\n{true_df['label'].value_counts().sort_index()}")

# 2. 检查LSTM预测
lstm_df = pd.read_csv('test_predictions_lstm.csv', header=None)
print("\n【LSTM预测 test_predictions_lstm.csv】")
print(f"样本数量: {len(lstm_df)}")
print(f"前20行:")
print(lstm_df.head(20))
print(f"\n预测值分布:\n{lstm_df[0].value_counts().sort_index()}")

# 3. 检查MLP预测
mlp_df = pd.read_csv('test_predictions_mlp.csv', header=None)
print("\n【MLP预测 test_predictions_mlp.csv】")
print(f"样本数量: {len(mlp_df)}")
print(f"前20行:")
print(mlp_df.head(20))
print(f"\n预测值分布:\n{mlp_df[0].value_counts().sort_index()}")

# 4. 如果有test_set.csv，也检查一下
try:
    test_set = pd.read_csv('test_set.csv')
    print("\n【test_set.csv】")
    print(f"样本数量: {len(test_set)}")
    print(f"列名: {test_set.columns.tolist()}")
    print(f"前10行:")
    print(test_set.head(10))
except:
    print("\n【test_set.csv】不存在")