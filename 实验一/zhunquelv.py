import pandas as pd
import numpy as np

# ==================== 加载数据 ====================

# 真实标签（111.csv）
true_labels_df = pd.read_csv('./111.csv')
true_labels = true_labels_df['label'].values

# ==================== LSTM预测结果 ====================
# test_predictions_lstm.csv 只有1列，按顺序排列
lstm_pred = pd.read_csv('./test_predictions_lstm.csv', header=None)
lstm_pred = lstm_pred.iloc[:, 0].values  # 取第一列

# ==================== MLP预测结果 ====================
# test_predictions_mlp.csv 只有1列，按顺序排列
mlp_pred = pd.read_csv('./test_predictions_mlp.csv', header=None)
mlp_pred = mlp_pred.iloc[:, 0].values  # 取第一列

# ==================== 按顺序对比 ====================

# 取最小长度对齐
n_samples = min(len(true_labels), len(lstm_pred), len(mlp_pred))
true_labels_aligned = true_labels[:n_samples]
lstm_pred_aligned = lstm_pred[:n_samples]
mlp_pred_aligned = mlp_pred[:n_samples]

# ==================== 计算准确率 ====================

# LSTM准确率
lstm_acc = (lstm_pred_aligned == true_labels_aligned).mean()
print(f"LSTM 准确率: {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
print(f"正确: {(lstm_pred_aligned == true_labels_aligned).sum()}/{n_samples}")

print("-" * 40)

# MLP准确率
mlp_acc = (mlp_pred_aligned == true_labels_aligned).mean()
print(f"MLP 准确率: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")
print(f"正确: {(mlp_pred_aligned == true_labels_aligned).sum()}/{n_samples}")

# ==================== 数据长度检查 ====================

print("\n" + "=" * 40)
print("数据长度检查:")
print(f"真实标签 (111.csv): {len(true_labels)} 个")
print(f"LSTM预测: {len(lstm_pred)} 个")
print(f"MLP预测: {len(mlp_pred)} 个")
print(f"实际对比样本数: {n_samples} 个")

if len(true_labels) != len(lstm_pred) or len(true_labels) != len(mlp_pred):
    print("\n⚠️ 注意: 样本数量不一致，只对比了前 {} 个".format(n_samples))