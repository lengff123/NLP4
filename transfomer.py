from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def load_stopwords(file_path):
    stop_words = []
    with open('cn_stopwords.txt', "r", encoding="utf-8", errors="ignore") as f:
        stop_words.extend([word.strip('\n') for word in f.readlines()])
    return stop_words

def preprocess_corpus( text,cn_stopwords):
    for tmp_char in cn_stopwords:
        text = text.replace(tmp_char, "")             
    return text 
merged_content = ''
stopwords_file_path = 'cn_stopwords.txt'
cn_stopwords = load_stopwords(stopwords_file_path)          
corpus_dict = {}  # 假设这是您的语料库字典
book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"#
for book_title in book_titles_list.split(','):
    book_title = book_title.strip()  # 去除可能存在的多余空白字符
    file_path='jyxstxtqj_downcc.com\{}.txt'.format(book_title)
    with open(file_path, 'r', encoding='utf-8') as f:
        merged_content += f.read()
    # 保存合并后的内容到新的文本文件
    merged_content=preprocess_corpus( merged_content,cn_stopwords)
output_file_path = '{}.txt'.format("all")
with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(merged_content)


# 使用GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 创建数据集
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="all.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 训练模型
training_args = TrainingArguments(
    output_dir="./gpt2_jin_yong",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

def generate_text_transformer(seed_text, next_words, model, tokenizer):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=next_words + len(input_ids[0]), num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_text_transformer("", 50, model, tokenizer))
