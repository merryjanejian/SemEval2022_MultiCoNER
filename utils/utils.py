from reader_utils import conll_to_df
from config import Config
from mertic_utils import compute_metrics

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import os
os.environ["WANDB_DISABLED"] = "true"

def tokenize_and_align_labels(examples,label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True )
    word_id_list=[]

    labels = []
    for i, label in enumerate(examples[f"{Config.task}_tags_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_id_list.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs['word_ids'] = word_id_list
    return tokenized_inputs


label_list = {0: 'O',1: 'B-PER',2: 'I-PER',3: 'B-GRP', 4: 'I-GRP',5: 'B-CW',6: 'I-CW',
 7: 'B-LOC',8: 'I-LOC',9:'B-CORP',10: 'I-CORP',11: 'B-PROD',12: 'I-PROD'}
ner_mapping={ key:id   for id,key in enumerate(label_list.keys())}

train_df = conll_to_df(file = f'./data/{Config.language}/{Config.language.split("-")[0].lower()}_train.conll')
train_df['ner_tags_ids'] = train_df['ner_tags'].apply(lambda x : [ ner_mapping[xx] for xx in x])
eval_df = conll_to_df(file = f'.//data/{Config.language}/{Config.language.split("-")[0].lower()}_dev.conll')
eval_df['ner_tags_ids'] = eval_df['ner_tags'].apply(lambda x : [ ner_mapping[xx] for xx in x])

train_datasets = Dataset.from_pandas(train_df)
eval_datasets = Dataset.from_pandas(eval_df)

#加载tokenizer
if 'roberta' in Config.model_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint,add_prefix_space=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint)

tokenized_train_datasets = train_datasets.map(tokenize_and_align_labels, batched=True)
tokenized_eval_datasets = eval_datasets.map(tokenize_and_align_labels, batched=True)

#加载模型
model = AutoModelForTokenClassification.from_pretrained(Config.model_checkpoint, num_labels=len(label_list))

#加载数据采集装置
data_collator = DataCollatorForTokenClassification(tokenizer)

#加载模型配置
model_name = Config.model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{Config.task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=25,
    weight_decay=0.01,
    push_to_hub=False,
    #save_steps=4000,
    save_strategy='epoch'
)

#初始化trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()


