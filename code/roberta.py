import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取CSV文件测试集
df = pd.read_csv('trace1000.csv')

X_train1= df['X1'].tolist()
X_train2 = df['X2'].tolist()
y_train= df['Y'].tolist()

# 读取CSV文件测试集
df = pd.read_csv('filtered_dataset.csv')

X_test1= df['X1'].tolist()
X_test2 = df['X2'].tolist()
y_test= df['Y'].tolist()


# # 将数据集分割为训练集和测试集
# X_train1, X_test1, X_train2, X_test2, y_train, y_test = train_test_split(X1, X2, Y, test_size=0.2)

# # 保存训练集和测试集为CSV文件
# train_df = pd.DataFrame({'X1': X_train1, 'X2': X_train2, 'Y': y_train})
# train_df.to_csv('train_all.csv', index=False)

# test_df = pd.DataFrame({'X1': X_test1, 'X2': X_test2, 'Y': y_test})
# test_df.to_csv('test_all.csv', index=False)


# 初始化Roberta tokenizer和model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#checkpoint_path = './results/checkpoint-1000'

# 加载模型，确保使用正确的路径
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,             # 训练轮数
    per_device_train_batch_size=16,  # 每个设备的训练批次大小
    per_device_eval_batch_size=64,   # 评估批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)

# 定义训练数据集
class TextSimilarityDataset:
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 将训练数据转换为模型可接受的格式
train_dataset = TextSimilarityDataset(tokenizer, X_train1 + X_train2, y_train)
eval_dataset = TextSimilarityDataset(tokenizer, X_test1 + X_test2, y_test)

# 定义评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = precision_recall_fscore_support(labels, predictions, average='binary')[-1]
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn= np.sum((predictions == 0) & (labels == 1))
    tn= np.sum((predictions == 0) & (labels == 0))
    return {'accuracy': accuracy, 'f1': f1, 'tp': tp, 'fp': fp,'fn':fn,'tn': tn}

# 重新初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  
)

# 评估模型
trainer.train()
results = trainer.evaluate()
print("Evaluation results:", results)