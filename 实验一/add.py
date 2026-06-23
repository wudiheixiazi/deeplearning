import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

# ==================== 创建picture文件夹 ====================
if not os.path.exists('./picture'):
    os.makedirs('./picture')
    print(f"✅ 已创建 picture 文件夹")

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 加载数据 ====================

# 1. 真实标签（111.csv）
true_labels_df = pd.read_csv('./111.csv')
print(f"真实标签数量: {len(true_labels_df)}")

# 2. 测试集（test_set.csv）
test_set_df = pd.read_csv('./test_set.csv')
print(f"测试集数量: {len(test_set_df)}")

# 3. LSTM预测结果
lstm_df = pd.read_csv('./test_predictions_lstm.csv', header=None)
lstm_pred = lstm_df.iloc[:, 0].values

# 4. MLP预测结果
mlp_df = pd.read_csv('./test_predictions_mlp.csv', header=None)
mlp_pred = mlp_df.iloc[:, 0].values

# ==================== 转换为数值类型 ====================

lstm_pred = pd.to_numeric(lstm_pred, errors='coerce')
mlp_pred = pd.to_numeric(mlp_pred, errors='coerce')

lstm_valid = ~np.isnan(lstm_pred)
mlp_valid = ~np.isnan(mlp_pred)

lstm_pred = lstm_pred[lstm_valid].astype(int)
mlp_pred = mlp_pred[mlp_valid].astype(int)

print(f"LSTM有效预测数: {len(lstm_pred)}")
print(f"MLP有效预测数: {len(mlp_pred)}")

# ==================== 给预测结果添加ID ====================

n = min(len(test_set_df), len(lstm_pred), len(mlp_pred))

lstm_result = pd.DataFrame({
    'id': test_set_df['id'].iloc[:n].values,
    'predicted_label': lstm_pred[:n]
})

mlp_result = pd.DataFrame({
    'id': test_set_df['id'].iloc[:n].values,
    'predicted_label': mlp_pred[:n]
})

# ==================== 保存带ID的预测结果 ====================

lstm_result.to_csv('./picture/test_predictions_lstm_with_id.csv', index=False)
mlp_result.to_csv('./picture/test_predictions_mlp_with_id.csv', index=False)

print(f"\n✅ 已保存带ID的预测结果到 picture 文件夹")

# ==================== 按ID合并计算准确率 ====================

merged_lstm = true_labels_df.merge(lstm_result, on='id', how='inner')
merged_mlp = true_labels_df.merge(mlp_result, on='id', how='inner')

lstm_acc = (merged_lstm['label'] == merged_lstm['predicted_label']).mean()
mlp_acc = (merged_mlp['label'] == merged_mlp['predicted_label']).mean()

lstm_correct = (merged_lstm['label'] == merged_lstm['predicted_label']).sum()
mlp_correct = (merged_mlp['label'] == merged_mlp['predicted_label']).sum()

# ==================== 输出准确率 ====================

print("\n" + "=" * 50)
print("模型准确率结果")
print("=" * 50)
print(f"LSTM 准确率: {lstm_acc:.4f} ({lstm_acc*100:.2f}%)")
print(f"      正确: {lstm_correct}/{len(merged_lstm)}")
print("-" * 50)
print(f"MLP  准确率: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")
print(f"      正确: {mlp_correct}/{len(merged_mlp)}")
print("=" * 50)

# ==================== 保存详细对比结果 ====================

lstm_compare = merged_lstm[['id', 'label', 'predicted_label']]
lstm_compare['correct'] = lstm_compare['label'] == lstm_compare['predicted_label']
lstm_compare.to_csv('./picture/lstm_detailed_comparison.csv', index=False)

mlp_compare = merged_mlp[['id', 'label', 'predicted_label']]
mlp_compare['correct'] = mlp_compare['label'] == mlp_compare['predicted_label']
mlp_compare.to_csv('./picture/mlp_detailed_comparison.csv', index=False)

print(f"\n✅ 详细对比结果已保存到 picture 文件夹")

# ==================== 获取所有类别 ====================
labels = sorted(set(merged_lstm['label']) | set(merged_lstm['predicted_label']))

# ==================== 图1: 准确率对比柱状图 ====================
fig1, ax1 = plt.subplots(figsize=(8, 6))
models = ['LSTM', 'MLP']
accuracies = [lstm_acc, mlp_acc]
colors = ['#3498db', '#e74c3c']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90%基准线')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./picture/accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/accuracy_comparison.png")

# ==================== 图2: LSTM混淆矩阵 ====================
fig2, ax2 = plt.subplots(figsize=(8, 7))
cm_lstm = confusion_matrix(merged_lstm['label'], merged_lstm['predicted_label'], labels=labels)
im2 = ax2.imshow(cm_lstm, cmap='Blues', interpolation='nearest')
ax2.set_xticks(np.arange(len(labels)))
ax2.set_yticks(np.arange(len(labels)))
ax2.set_xticklabels([f'{int(l)}' for l in labels])
ax2.set_yticklabels([f'{int(l)}' for l in labels])
ax2.set_xlabel('预测标签', fontsize=12)
ax2.set_ylabel('真实标签', fontsize=12)
ax2.set_title(f'LSTM 混淆矩阵\n准确率: {lstm_acc*100:.2f}%', fontsize=14, fontweight='bold')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax2.text(j, i, f'{cm_lstm[i, j]}', ha='center', va='center', 
                 color='white' if cm_lstm[i, j] > cm_lstm.max()/2 else 'black', fontsize=11)
plt.colorbar(im2, ax=ax2, label='样本数')
plt.tight_layout()
plt.savefig('./picture/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/lstm_confusion_matrix.png")

# ==================== 图3: MLP混淆矩阵 ====================
fig3, ax3 = plt.subplots(figsize=(8, 7))
cm_mlp = confusion_matrix(merged_mlp['label'], merged_mlp['predicted_label'], labels=labels)
im3 = ax3.imshow(cm_mlp, cmap='Reds', interpolation='nearest')
ax3.set_xticks(np.arange(len(labels)))
ax3.set_yticks(np.arange(len(labels)))
ax3.set_xticklabels([f'{int(l)}' for l in labels])
ax3.set_yticklabels([f'{int(l)}' for l in labels])
ax3.set_xlabel('预测标签', fontsize=12)
ax3.set_ylabel('真实标签', fontsize=12)
ax3.set_title(f'MLP 混淆矩阵\n准确率: {mlp_acc*100:.2f}%', fontsize=14, fontweight='bold')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax3.text(j, i, f'{cm_mlp[i, j]}', ha='center', va='center',
                 color='white' if cm_mlp[i, j] > cm_mlp.max()/2 else 'black', fontsize=11)
plt.colorbar(im3, ax=ax3, label='样本数')
plt.tight_layout()
plt.savefig('./picture/mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/mlp_confusion_matrix.png")

# ==================== 图4: 各类别准确率对比 ====================
fig4, ax4 = plt.subplots(figsize=(10, 6))
class_acc_lstm = {}
class_acc_mlp = {}
for label in labels:
    mask = (merged_lstm['label'] == label)
    class_acc_lstm[label] = (merged_lstm.loc[mask, 'label'] == merged_lstm.loc[mask, 'predicted_label']).mean() if mask.sum() > 0 else 0
    
    mask = (merged_mlp['label'] == label)
    class_acc_mlp[label] = (merged_mlp.loc[mask, 'label'] == merged_mlp.loc[mask, 'predicted_label']).mean() if mask.sum() > 0 else 0

x = np.arange(len(labels))
width = 0.35
bars1 = ax4.bar(x - width/2, [class_acc_lstm[l] for l in labels], width, label='LSTM', color='#3498db', edgecolor='black')
bars2 = ax4.bar(x + width/2, [class_acc_mlp[l] for l in labels], width, label='MLP', color='#e74c3c', edgecolor='black')
ax4.set_xlabel('类别', fontsize=12)
ax4.set_ylabel('类别准确率', fontsize=12)
ax4.set_title('各类别准确率对比', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'类别 {int(l)}' for l in labels])
ax4.legend()
ax4.set_ylim(0, 1.05)
ax4.grid(axis='y', alpha=0.3)
# 在柱子上显示数值
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('./picture/class_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/class_accuracy_comparison.png")

# ==================== 图5: 预测分布对比 ====================
fig5, ax5 = plt.subplots(figsize=(10, 6))
true_counts = merged_lstm['label'].value_counts().sort_index()
lstm_counts = merged_lstm['predicted_label'].value_counts().sort_index()
mlp_counts = merged_mlp['predicted_label'].value_counts().sort_index()

x = np.arange(len(labels))
width = 0.25
ax5.bar(x - width, [true_counts.get(l, 0) for l in labels], width, label='真实标签', color='#2ecc71', edgecolor='black')
ax5.bar(x, [lstm_counts.get(l, 0) for l in labels], width, label='LSTM预测', color='#3498db', edgecolor='black')
ax5.bar(x + width, [mlp_counts.get(l, 0) for l in labels], width, label='MLP预测', color='#e74c3c', edgecolor='black')
ax5.set_xlabel('类别', fontsize=12)
ax5.set_ylabel('样本数量', fontsize=12)
ax5.set_title('预测分布对比', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([f'类别 {int(l)}' for l in labels])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./picture/prediction_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/prediction_distribution.png")

# ==================== 图6: 正确/错误样本统计 ====================
fig6, ax6 = plt.subplots(figsize=(8, 6))
lstm_errors = (merged_lstm['label'] != merged_lstm['predicted_label']).sum()
mlp_errors = (merged_mlp['label'] != merged_mlp['predicted_label']).sum()
total = len(merged_lstm)

correct_data = [total - lstm_errors, total - mlp_errors]
error_data = [lstm_errors, mlp_errors]
x = np.arange(len(models))
width = 0.5

ax6.bar(x - width/2, correct_data, width, label='正确', color='#2ecc71', edgecolor='black')
ax6.bar(x + width/2, error_data, width, label='错误', color='#e74c3c', edgecolor='black')

ax6.set_xlabel('模型', fontsize=12)
ax6.set_ylabel('样本数量', fontsize=12)
ax6.set_title('正确/错误样本统计', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(models)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

for i, (correct, error) in enumerate(zip(correct_data, error_data)):
    ax6.text(i - width/2, correct + 5, f'{correct}', ha='center', va='bottom', fontsize=10)
    ax6.text(i + width/2, error + 5, f'{error}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('./picture/correct_error_stats.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存: picture/correct_error_stats.png")

# ==================== 打印分类报告 ====================

print("\n" + "=" * 50)
print("LSTM 分类报告")
print("=" * 50)
print(classification_report(merged_lstm['label'], merged_lstm['predicted_label'], 
                           target_names=[f'类别 {int(l)}' for l in labels], digits=4))

print("\n" + "=" * 50)
print("MLP 分类报告")
print("=" * 50)
print(classification_report(merged_mlp['label'], merged_mlp['predicted_label'],
                           target_names=[f'类别 {int(l)}' for l in labels], digits=4))

# ==================== 显示错误样本 ====================

lstm_errors_df = merged_lstm[merged_lstm['label'] != merged_lstm['predicted_label']]
mlp_errors_df = merged_mlp[merged_mlp['label'] != merged_mlp['predicted_label']]

print("\n" + "=" * 50)
print("错误样本统计")
print("=" * 50)
print(f"LSTM 错误样本数: {len(lstm_errors_df)}/{len(merged_lstm)} ({len(lstm_errors_df)/len(merged_lstm)*100:.2f}%)")
print(f"MLP  错误样本数: {len(mlp_errors_df)}/{len(merged_mlp)} ({len(mlp_errors_df)/len(merged_mlp)*100:.2f}%)")

if len(lstm_errors_df) > 0:
    print(f"\nLSTM 错误样本前10个:")
    print(lstm_errors_df[['id', 'label', 'predicted_label']].head(10))
    # 保存错误样本到CSV
    lstm_errors_df.to_csv('./picture/lstm_error_samples.csv', index=False)
    print("✅ 已保存: picture/lstm_error_samples.csv")
    
if len(mlp_errors_df) > 0:
    print(f"\nMLP 错误样本前10个:")
    print(mlp_errors_df[['id', 'label', 'predicted_label']].head(10))
    mlp_errors_df.to_csv('./picture/mlp_error_samples.csv', index=False)
    print("✅ 已保存: picture/mlp_error_samples.csv")

print("\n" + "=" * 50)
print("所有图片已保存到 picture 文件夹:")
print("  1. accuracy_comparison.png - 准确率对比")
print("  2. lstm_confusion_matrix.png - LSTM混淆矩阵")
print("  3. mlp_confusion_matrix.png - MLP混淆矩阵")
print("  4. class_accuracy_comparison.png - 各类别准确率对比")
print("  5. prediction_distribution.png - 预测分布对比")
print("  6. correct_error_stats.png - 正确/错误样本统计")
print("=" * 50)