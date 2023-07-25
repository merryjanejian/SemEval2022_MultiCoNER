from utils.reader_utils import conll_to_df
from config import Config
from utils.mertic_utils import compute_metrics
from tokenize_and_align import tokenize_and_align_labels
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import os
os.environ["WANDB_DISABLED"] = "true"

def get_data(Config:object,type='train'):
    df = conll_to_df(file = f'./data/{Config.language}/{Config.language.split("-")[0].lower()}_{type}.conll')
    df['ner_tags_ids'] = df['ner_tags'].apply(lambda x : [ Config.ner_mapping[xx] for xx in x])
    datasets =  Dataset.from_pandas(df)
    return datasets

def train():

    #step1 获取训练数据
    train_datasets = get_data(Config,'train')
    eval_datasets = get_data(Config,'dev')

    #step2 加载tokenizer
    if 'roberta' in Config.model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint,add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

    #step3 将数据tokenizing
    tokenized_train_datasets = train_datasets.map(partial(tokenize_and_align_labels,tokenizer=tokenizer), batched=True)
    tokenized_eval_datasets = eval_datasets.map(partial(tokenize_and_align_labels,tokenizer=tokenizer), batched=True)
    #step4.1 加载模型
    model = AutoModelForTokenClassification.from_pretrained(Config.model_checkpoint, num_labels=len(Config.label_list))

    #step4.2加载数据采集装置
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #step4.3加载模型配置
    model_name = Config.model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-{Config.task}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        num_train_epochs=25,
        weight_decay=0.01,
        push_to_hub=False,
        #save_steps=4000,
        save_strategy='epoch'
    )

    #step5 初始化trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_eval_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    #step5.1 训练
    trainer.train()
    trainer.evaluate()

if __name__=="__main__":
    train()


