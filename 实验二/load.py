# 【服务器必备】置顶配置：HF镜像 + 绘图后端
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子（和训练时保持一致）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径
SAVE_DIR = './saved_models'
MODEL_PATH = os.path.join(SAVE_DIR, 'best_bert_model.pth')
TRAIN_DATA_PATH = 'train_set_8.csv'
TEST_DATA_PATH = 'test_set_2.csv'
MAX_LEN = 128
BATCH_SIZE = 32

# ==================== 1. 加载数据（和训练时保持一致） ====================
print("\n" + "=" * 70)
print("加载数据")
print("=" * 70)

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# 标签处理（和训练时保持一致）
if train_df['label'].min() == 1:
    print("\n检测到标签从1开始，转换为0-based...")
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1
elif train_df['label'].min() == 0:
    print("\n标签已经是0-based，无需转换")

num_classes = train_df['label'].nunique()
print(f"类别数: {num_classes}")

# ==================== 2. 加载BERT分词器和模型 ====================
print("\n" + "=" * 70)
print("加载BERT分词器和模型结构")
print("=" * 70)

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# ==================== 3. 定义Dataset和模型结构（与训练时完全一致） ====================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 初始化分类层（与训练时保持一致）
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ==================== 4. 加载保存的模型 ====================
print("\n" + "=" * 70)
print("加载保存的模型")
print("=" * 70)

# 创建模型（使用与训练时相同的dropout参数）
dropout_rate = 0.3  # 与训练时保持一致
model = BERTClassifier(bert_model, num_classes, dropout=dropout_rate)
model = model.to(device)

# 加载checkpoint
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载模型成功")
    print(f"最佳准确率: {checkpoint['test_acc']:.4f}")
    print(f"对应Epoch: {checkpoint['epoch'] + 1}")
except FileNotFoundError:
    print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
    print("请确保先运行训练脚本生成模型")
    exit(1)
except KeyError as e:
    print(f"❌ 错误：checkpoint格式不正确，缺少键 {e}")
    print("请检查训练脚本保存的checkpoint格式")
    exit(1)

# ==================== 5. 创建DataLoader ====================
print("\n" + "=" * 70)
print("创建DataLoader")
print("=" * 70)

test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer, MAX_LEN)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,  # 设置为0避免多进程问题
    pin_memory=True if device.type == 'cuda' else False
)
print(f"测试集样本数: {len(test_dataset)}")
print(f"测试集批次数: {len(test_loader)}")

# ==================== 6. 评估函数 ====================
def evaluate(model, loader, criterion):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print("\n开始评估测试集...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估进度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy, all_preds, all_labels

# ==================== 7. 重新评估并生成结果 ====================
print("\n" + "=" * 70)
print("重新评估模型")
print("=" * 70)

criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

print(f"\n📊 评估结果:")
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"最佳模型准确率: {checkpoint['test_acc']:.4f}")

# 生成分类报告
print("\n详细分类报告:")
report = classification_report(test_labels, test_preds, digits=4)
print(report)

# 保存分类报告
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("BERT分类模型结果（加载自保存模型）\n")
    f.write("=" * 50 + "\n")
    f.write(f"训练时的最佳准确率: {checkpoint['test_acc']:.4f}\n")
    f.write(f"重新评估准确率: {test_acc:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)
print(f"\n✅ 分类报告已保存到 {SAVE_DIR}/classification_report.txt")

# ==================== 8. 生成可视化图片 ====================
print("\n" + "=" * 70)
print("开始生成可视化图片")
print("=" * 70)

try:
    # 图1：混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix (Accuracy: {test_acc:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存到 {SAVE_DIR}/confusion_matrix.png")

    # 图2：性能概览图
    plt.figure(figsize=(8, 6))
    metrics = ['Accuracy', 'Loss (scaled)']
    # 将损失缩放到0-1范围以便可视化
    scaled_loss = max(0, min(1, 1 - test_loss))  # 损失越小越好
    values = [test_acc, scaled_loss]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylim(0, 1.1)
    plt.title('Model Test Performance', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, [test_acc, test_loss]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能概览图已保存到 {SAVE_DIR}/model_performance.png")
    
    # 图3：各类别准确率
    class_correct = []
    for i in range(num_classes):
        mask = np.array(test_labels) == i
        if mask.sum() > 0:
            acc = np.mean(np.array(test_preds)[mask] == i)
        else:
            acc = 0
        class_correct.append(acc)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(num_classes), class_correct, color='lightgreen', alpha=0.7)
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, class_correct):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 各类别准确率图已保存到 {SAVE_DIR}/per_class_accuracy.png")

except Exception as e:
    print(f"❌ 生成图片失败: {e}")
    print("请执行安装依赖：pip install matplotlib seaborn")

# ==================== 9. 显示示例预测 ====================
print("\n" + "=" * 70)
print("示例预测（前10个测试样本）")
print("=" * 70)

model.eval()
sample_texts = test_df['text'].head(10).values
sample_labels = test_df['label'].head(10).values

correct_predictions = 0
for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
    # 分词
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=1).item()
    
    status = "✓" if pred == true_label else "✗"
    if pred == true_label:
        correct_predictions += 1
    
    print(f"\n{status} 样本 {i+1}:")
    print(f"   文本预览: {text[:80]}...")
    print(f"   真实标签: {true_label}")
    print(f"   预测标签: {pred}")

print(f"\n前10个样本预测准确率: {correct_predictions}/10 = {correct_predictions/10:.1%}")

print("\n" + "=" * 70)
print("✅ 所有流程执行完毕！")
print("=" * 70)