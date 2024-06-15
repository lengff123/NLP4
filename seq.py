import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import random


def load_stopwords(file_path):
    stop_words = []
    with open('cn_stopwords.txt', "r", encoding="utf-8", errors="ignore") as f:
        stop_words.extend([word.strip('\n') for word in f.readlines()])
    return stop_words

def preprocess_corpus( text,cn_stopwords):
    for tmp_char in cn_stopwords:
        text = text.replace(tmp_char, "")             
    return text 


if __name__ == '__main__':
    stopwords_file_path = 'cn_stopwords.txt'
    cn_stopwords = load_stopwords(stopwords_file_path)          
    corpus_dict = {}  # 假设这是您的语料库字典
    book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"#
    for book_title in book_titles_list.split(','):
        book_title = book_title.strip()  # 去除可能存在的多余空白字符
        file_path='jyxstxtqj_downcc.com\{}.txt'.format(book_title)
        merged_content = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            merged_content += f.read()
        # 保存合并后的内容到新的文本文件
        merged_content=preprocess_corpus( merged_content,cn_stopwords)
        output_file_path = 'fr\{}.txt'.format(book_title)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
            
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
import re

# 读取语料库
with open('jin_yong_corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

# 分句
sentences = corpus.split('\n')

# 使用Counter进行分词
tokenizer = Counter(re.findall(r'\b\w+\b', corpus))
word_index = {word: idx+1 for idx, (word, _) in enumerate(tokenizer.items())}
total_words = len(word_index) + 1

# 创建输入和输出序列
input_sequences = []
for sentence in sentences:
    token_list = [word_index[word] for word in re.findall(r'\b\w+\b', sentence)]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 填充序列
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequence([torch.tensor(seq) for seq in input_sequences], batch_first=True, padding_value=0)

# 创建训练数据
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = torch.nn.functional.one_hot(labels, num_classes=total_words).float()

class TextDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

dataset = TextDataset(xs, ys)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class TextGenerationModel(nn.Module):
    def __init__(self, total_words, embed_dim, hidden_dim):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(total_words, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, total_words)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = TextGenerationModel(total_words, 64, 20)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def generate_text_seq2seq(seed_text, next_words, model, max_sequence_len):
    model.eval()
    words = re.findall(r'\b\w+\b', seed_text)
    for _ in range(next_words):
        token_list = [word_index[word] for word in words]
        token_list = torch.tensor(token_list).unsqueeze(0)
        token_list = pad_sequence([token_list], batch_first=True, padding_value=0).to(dtype=torch.long)
        token_list = token_list[:, -max_sequence_len+1:]
        predicted = model(token_list)
        predicted = torch.argmax(predicted, axis=-1).item()
        output_word = list(word_index.keys())[list(word_index.values()).index(predicted)]
        words.append(output_word)
    return ' '.join(words)

print(generate_text_seq2seq("", 50, model, max_sequence_len))



        
