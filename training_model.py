"""
Code adapted from a jupyter notebook.
"""
import json

import numpy as np
import pandas as pd

from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average="micro")


def precision_multiclass(labels, preds):
    return precision_score(labels, preds, average="micro")


def recall_multiclass(labels, preds):
    return recall_score(labels, preds, average="micro")


puc_train = pd.read_json(
    "storage/2020-09-12_puc_augmented.json", orient="records")
puc_test = pd.read_json("storage/puc_test_30.json", orient="records")

puj_train = pd.read_json(
    "storage/2020-09-12_puj_augmented.json", orient="records")
puj_test = pd.read_json("storage/puj_test_30.json", orient="records")

aurora_train = pd.read_json(
    "storage/2020-09-12_aurora_augmented.json", orient="records")
aurora_test = pd.read_json("storage/aurora_test_30.json", orient="records")

train = pd.concat([puc_train, puj_train, aurora_train], ignore_index=True)
test = pd.concat([puc_test, puj_test, aurora_test], ignore_index=True)

train["first_sdg"] = train["first_sdg"] - 1
test["first_sdg"] = test["first_sdg"] - 1

train = train[["clean_abstract", "first_sdg"]]
test = test[["clean_abstract", "first_sdg"]]

print("Data loaded successfully.")


model_args = {
    "learning_rate": 1e-5,
    "num_train_epochs": 2,
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "max_seq_length": 512
}

model = ClassificationModel(
    "roberta", "roberta-large", num_labels=len(train["first_sdg"].unique()), args=model_args
)
model.train_model(train)
print("Training has finished.")

results, model_outputs, predictions = model.eval_model(
    test, acc=accuracy_score, f1=f1_multiclass, precision=precision_multiclass, recall=recall_multiclass)
predictions = np.argmax(model_outputs, axis=-1).tolist()
print(results)

with open("storage/2020-09-16_roberta_results.json", "w") as f:
    json.dump(results, f)
with open("storage/2020-09-16_roberta_predictions.json", "w") as f:
    json.dump(predictions, f)
