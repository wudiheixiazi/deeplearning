import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置环境变量用于调试（可选）
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("警告: 没有检测到GPU，将使用CPU训练（会很慢）")

# ==================== 1. 数据加载 ====================
print("\n" + "=" * 70)
print("加载数据")
print("=" * 70)

# 使用当前路径
train_df = pd.read_csv('train_set_8.csv')
test_df = pd.read_csv('test_set_2.csv')

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")
print(f"原始标签类别数: {train_df['label'].nunique()}")
print(f"\n训练集标签分布:\n{train_df['label'].value_counts().sort_index()}")
print(f"\n测试集标签分布:\n{test_df['label'].value_counts().sort_index()}")

# 检查标签范围
print(f"\n训练集标签范围: [{train_df['label'].min()}, {train_df['label'].max()}]")
print(f"测试集标签范围: [{test_df['label'].min()}, {test_df['label'].max()}]")

# 标签从0开始（BERT需要）
# 注意：如果标签已经是0-based，就不需要减1
if train_df['label'].min() == 1:
    print("\n检测到标签从1开始，转换为0-based...")
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1
elif train_df['label'].min() == 0:
    print("\n标签已经是0-based，无需转换")
else:
    print(f"\n警告: 标签最小值是 {train_df['label'].min()}，请检查！")

num_classes = train_df['label'].nunique()
print(f"\n类别数: {num_classes}")
print(f"转换后标签范围: [0, {num_classes-1}]")

# 验证标签没有负数或超出范围
if train_df['label'].min() < 0 or train_df['label'].max() >= num_classes:
    print(f"错误: 标签超出范围！")
    print(f"最小值: {train_df['label'].min()}")
    print(f"最大值: {train_df['label'].max()}")
    print(f"有效范围: [0, {num_classes-1}]")
    exit(1)

# ==================== 2. 加载BERT模型和分词器 ====================
print("\n" + "=" * 70)
print("加载BERT模型和分词器")
print("=" * 70)

# 使用中文BERT
model_name = 'bert-base-chinese'

# 尝试使用镜像（如果是国内网络）
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    print("BERT模型加载成功")
except Exception as e:
    print(f"加载失败: {e}")
    print("尝试使用本地缓存或镜像...")
    # 如果是网络问题，可以手动指定本地路径
    # tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    # bert_model = BertModel.from_pretrained('./bert-base-chinese')
    raise

print(f"词表大小: {tokenizer.vocab_size}")

# ==================== 3. 创建Dataset ====================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.values
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = len(np.unique(labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 验证标签
        if label < 0 or label >= self.num_classes:
            print(f"警告: 索引 {idx} 的标签 {label} 无效，有效范围 [0, {self.num_classes-1}]")
            label = max(0, min(label, self.num_classes - 1))

        # 分词
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

# 参数设置
MAX_LEN = 128
BATCH_SIZE = 32

# 创建数据集
train_dataset = NewsDataset(train_df['text'], train_df['label'], tokenizer, MAX_LEN)
test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer, MAX_LEN)

# 使用更稳定的DataLoader配置
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,
    pin_memory=True if device.type == 'cuda' else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True if device.type == 'cuda' else False
)

print(f"\n训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ==================== 4. 定义BERT分类模型 ====================
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 初始化分类层
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]标记的输出
        pooled_output = outputs.pooler_output
        # Dropout和分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 创建模型
model = BERTClassifier(bert_model, num_classes, dropout=0.3)
model = model.to(device)

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 5. 训练配置 ====================
EPOCHS = 5
LEARNING_RATE = 2e-5

# 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 学习率调度器
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 混合精度训练
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

print(f"\n训练配置:")
print(f"  训练轮数: {EPOCHS}")
print(f"  学习率: {LEARNING_RATE}")
print(f"  批大小: {BATCH_SIZE}")
print(f"  总步数: {total_steps}")
print(f"  预热步数: {warmup_steps}")
print(f"  混合精度: {'启用' if scaler else '禁用'}")

# ==================== 6. 训练函数（带调试） ====================
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        # 将数据移到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 验证标签（只在第一个batch打印）
        if batch_idx == 0:
            print(f"\n标签范围: [{labels.min().item()}, {labels.max().item()}]")
            print(f"唯一标签值: {torch.unique(labels).tolist()}")
            if labels.min() < 0 or labels.max() >= num_classes:
                print(f"错误: 标签超出范围！")
                print(f"最小值: {labels.min().item()}")
                print(f"最大值: {labels.max().item()}")
                print(f"有效范围: [0, {num_classes-1}]")
                raise ValueError("标签超出范围")

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

        # 记录损失
        total_loss += loss.item()

        # 记录预测结果
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})

        # 调试：检查前几个batch
        if batch_idx == 0:
            print(f"第一个batch - 损失: {loss.item():.4f}")
            print(f"预测样本: {preds[:5].tolist()}")
            print(f"真实标签: {labels[:5].tolist()}")

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy

# ==================== 7. 评估函数 ====================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
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

# ==================== 8. 训练循环 ====================
print("\n" + "=" * 70)
print("开始训练")
print("=" * 70)

best_test_acc = 0
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# 创建保存模型的目录
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    # 训练
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch+1)
    train_time = time.time() - start_time

    # 评估
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

    # 记录
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 时间: {train_time:.2f}s")
    print(f"测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

    # 保存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'num_classes': num_classes,
        }, os.path.join(save_dir, 'best_bert_model.pth'))
        print(f"✓ 保存最佳模型 (准确率: {test_acc:.4f})")

# ==================== 9. 最终评估 ====================
print("\n" + "=" * 70)
print("最终评估结果")
print("=" * 70)

# 加载最佳模型
checkpoint = torch.load(os.path.join(save_dir, 'best_bert_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"加载最佳模型 (Epoch {checkpoint['epoch'] + 1}, 准确率: {checkpoint['test_acc']:.4f})")

# 最终评估
_, final_acc, final_preds, final_labels = evaluate(model, test_loader, criterion)

print(f"\n最终测试集准确率: {final_acc:.4f}")
print("\n详细分类报告:")
print(classification_report(final_labels, final_preds, digits=4))

# 保存分类报告到文件
with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write("BERT分类模型结果\n")
    f.write("=" * 50 + "\n")
    f.write(f"最佳测试集准确率: {best_test_acc:.4f}\n")
    f.write(f"最终测试集准确率: {final_acc:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(classification_report(final_labels, final_preds, digits=4))

# 保存模型（最终版本）
torch.save(model.state_dict(), os.path.join(save_dir, 'bert_model_final.pth'))
print(f"\n模型已保存到: {save_dir}/bert_model_final.pth")

# ==================== 10. 预测示例 ====================
print("\n" + "=" * 70)
print("预测示例")
print("=" * 70)

model.eval()
sample_texts = test_df['text'].head(10).values
sample_labels = test_df['label'].head(10).values

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
    print(f"\n{status} 样本 {i + 1}:")
    print(f"   文本: {text[:100]}...")
    print(f"   真实标签: {true_label + 1 if true_label >= 0 else true_label}")  # 恢复原始标签显示
    print(f"   预测标签: {pred + 1 if pred >= 0 else pred}")

print("\n" + "=" * 70)
print("训练完成！")
print("=" * 70)

# 绘制训练曲线（可选）
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    print(f"\n训练曲线已保存到 {save_dir}/training_curves.png")
except:
    print("\n注意: matplotlib未安装，跳过绘制训练曲线")