#-------------------------------------------------
#预训练模型 采用WWM的方式
#-------------------------------------------------
import os
os.environ["WANDB_DISABLED"] = "true"

encoder_model = "D:\\AI-NLP-Jane\\mycode\\base\\bert-base-uncased"#"bert-base-uncased"
language = 'EN-English'
pretrain_file = f'./data/{language}/{language.split("-")[0].lower()}_train.csv'
model_name = encoder_model.split("\\")[-1]


from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForPreTraining
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers import BertForMaskedLM,BertForPreTraining,BertTokenizer


tokenizer = BertTokenizer.from_pretrained(encoder_model)
model = BertForMaskedLM.from_pretrained(encoder_model)
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=pretrain_file,
    block_size=128,
)
wwm_data_collator = DataCollatorForWholeWordMask(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )
training_args = TrainingArguments(
    output_dir= f"{model_name}-wwm-pretrain",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_total_limit=2,
    seed=1,
    #evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    #save_steps=4000,
    #save_strategy='epoch',
    #prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=wwm_data_collator,
    train_dataset=dataset
)
trainer.train()