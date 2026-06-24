import os
import re
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

picture_dir = "picture"
if not os.path.exists(picture_dir):
    os.makedirs(picture_dir)

# 从分类报告中读取准确率
def get_accuracy_from_report(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        # 查找准确率，例如 "accuracy: 0.9652" 或 "accuracy                         0.97"
        match = re.search(r'accuracy\s*:?\s*(\d+\.?\d*)', content, re.IGNORECASE)
        if match:
            acc = float(match.group(1)) * 100 if float(match.group(1)) < 1 else float(match.group(1))
            return acc
    return None

# 读取BERT准确率
bert_accuracy = get_accuracy_from_report("saved_models/classification_report.txt")
if bert_accuracy is None:
    bert_accuracy = 96.5  # 默认值，请手动修改
    print(f"警告: 未找到分类报告，使用默认值 {bert_accuracy}%")

# 读取MLP准确率（如果有的话）
mlp_accuracy = get_accuracy_from_report("saved_models/mlp_classification_report.txt")
if mlp_accuracy is None:
    mlp_accuracy = 86.5  # 默认值，请手动修改
    print(f"警告: 未找到MLP分类报告，使用默认值 {mlp_accuracy}%")

print(f"BERT模型准确率: {bert_accuracy}%")
print(f"MLP模型准确率: {mlp_accuracy}%")

# 画图1
plt.figure(figsize=(6, 5))
plt.bar(["BERT"], [bert_accuracy], color="#1f77b4", width=0.5)
plt.text(0, bert_accuracy + 1, f'{bert_accuracy}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.ylim(0, 105)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("BERT Model Prediction Accuracy", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, "bert_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()

# 画图2
plt.figure(figsize=(6, 5))
plt.bar(["MLP"], [mlp_accuracy], color="#ff7f0e", width=0.5)
plt.text(0, mlp_accuracy + 1, f'{mlp_accuracy}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.ylim(0, 105)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("MLP Model Prediction Accuracy", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, "mlp_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n图片已保存到 picture/ 文件夹")