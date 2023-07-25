from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


def compute_metrics(p):
    '''

    Args:
        p:EvalPrediction(predictions=all_preds, label_ids=all_labels)
    注意：label_list 和 eval_tokenized_datasets 作为全局变量
    也可以改trainner.evaluation_loop函数
    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    把eval_dataset 传进去，这样就可以用到 word_ids信息
    Returns:

    '''
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    predict_tags = []
    true_tags = []
    for idx in range(len(eval_tokenized_datasets)):
        predict_tag = get_tags_by_wordid(eval_tokenized_datasets[idx]['word_ids'][1:-1], true_predictions[idx])
        true_tag = get_tags_by_wordid(eval_tokenized_datasets[idx]['word_ids'][1:-1], true_labels[idx])
        predict_tags.append(predict_tag)
        true_tags.append(true_tag)

    return {
        "precision": precision_score(predict_tags, true_tags),
        "recall": recall_score(predict_tags, true_tags),
        "f1": f1_score(predict_tags, true_tags),
        "accuracy": accuracy_score(predict_tags, true_tags)
    }

def get_tags_by_wordid(word_ids, tags):
    tags_results = []
    assert len(word_ids) == len(tags)
    pre_id = -1
    for word_id, tag_id in zip(word_ids, range(len(tags))):
        if word_id != pre_id:
            tags_results.append(tags[tag_id])
        pre_id = word_id

    return tags_results