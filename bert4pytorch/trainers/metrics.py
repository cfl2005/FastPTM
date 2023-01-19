from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report as sl_classification_report

def compute_cls_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}

    classification_report_dict = classification_report(labels, preds, output_dict=True)

    for key0, val0 in classification_report_dict.items():
        if key0 == 'weighted avg':
            if isinstance(val0, dict):
                for key1, val1 in val0.items():
                    if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
                        results[key0 + "__" + key1] = val1
            else:
                results[key0] = val0

    accuracy = accuracy_score(labels, preds)
    results['accuracy'] = accuracy
    return results


def compute_sequence_labeling_metrics(preds, labels):
    assert len(preds) == len(labels)
    results = {}

    classification_report_dict = sl_classification_report(labels, preds, output_dict=True)
    for key0, val0 in classification_report_dict.items():
        if key0 == 'weighted avg':
            if isinstance(val0, dict):
                for key1, val1 in val0.items():
                    if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
                        results[key0 + "__" + key1] = val1
            else:
                results[key0] = val0

    return results
