
class Config():
    task = "ner"  # Should be one of "ner", "pos" or "chunk"
    model_checkpoint = "D:/AI-NLP-Jane/mycode/base/bert-base-uncased"
        #"bert-large-uncased-whole-word-masking"#"bert-base-uncased"  # 英文baseline
    batch_size = 16
    language = 'EN-English'
    label_list = {0: 'O',1: 'B-PER',2: 'I-PER',3: 'B-GRP', 4: 'I-GRP',5: 'B-CW',6: 'I-CW',
     7: 'B-LOC',8: 'I-LOC',9:'B-CORP',10: 'I-CORP',11: 'B-PROD',12: 'I-PROD'}
    ner_mapping={ key:id   for id,key in label_list.items()}