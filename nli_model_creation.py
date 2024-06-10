import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.utils import logging

from datasets import load_dataset, load_metric, DatasetDict, concatenate_datasets
from pprint import pprint

from utils.functions import remove_outliers_from_datasets

assert torch.cuda.is_available()

datasets_names = sys.argv[1]
datasets_arg = sys.argv[2]

datasets_list = []

if "xnli_fr" in datasets_names:
    datasets_list.append(load_dataset("./datasets/xnli_fr", datasets_arg, trust_remote_code=True))
if "repnum_wl" in datasets_names:
    datasets_list.append(remove_outliers_from_datasets(load_dataset("./datasets/repnum_nli", datasets_arg, trust_remote_code=True)))
if "rua_wl" in datasets_names:
    datasets_list.append(remove_outliers_from_datasets(load_dataset("./datasets/rua_nli", datasets_arg, trust_remote_code=True)))

train_dataset = concatenate_datasets([dataset["train"] for dataset in datasets_list])
eval_dataset = concatenate_datasets([dataset["validation"] for dataset in datasets_list])
test_dataset = concatenate_datasets([dataset["test"] for dataset in datasets_list])

nli_datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset}).shuffle(seed=1234)

model_checkpoint = sys.argv[3]
model_name = model_checkpoint.split("/")[-1]
batch_size = 8

trainer_name = f"{model_name}-finetuned-{datasets_names}{('_' + datasets_arg) if datasets_arg != '2_classes' else ''}"

label_list = nli_datasets["train"].features["label"].names

config = AutoConfig.from_pretrained(model_checkpoint)

config.id2label = {idx: label for (idx, label) in enumerate(label_list)}
config.label2id = {label: idx for (idx, label) in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512, use_fast=True)

encoded_dataset = nli_datasets.map(lambda examples: tokenizer(examples["premise"], examples["hypothesis"], max_length=512, truncation=True), batched=True)

num_labels = len(label_list)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

metric_name = "f1"
metric = load_metric(metric_name)

model.config.name_or_path = f"waboucay/{trainer_name}"


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "f1_micro": metric.compute(predictions=predictions, references=labels, average="micro")["f1"],
        "f1_macro": metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }


args = TrainingArguments(
    f"../scratch/{trainer_name}",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_safetensors=True,
    save_total_limit=3
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logging.set_verbosity_info()

trainer.train()
print("With validation set:")
pprint(trainer.evaluate())
print("With test set:")
pprint(trainer.evaluate(eval_dataset=encoded_dataset["test"]))

trainer.save_model(trainer_name)

log_history = trainer.state.log_history
x = sorted(list({log["step"] for log in log_history}))
y1 = [log["loss"] if "loss" in log else log["train_loss"] for log in list(filter(lambda log: ("loss" in log) or ("train_loss" in log), log_history))]
y2 = [log["eval_loss"] for log in list(filter(lambda log: "eval_loss" in log, log_history))]

if len(x) < len(y1) or len(x) < len(y2):
    print(f"log_history: {log_history}")
    y1 = y1[:len(x)]
    y2 = y2[:len(x)]

fig, ax = plt.subplots()
ax.plot(x, y1, 'r', label="train_loss")
ax.plot(x, y2, 'g', label="eval_loss")
ax.set_ylim(0, 1)
ax.set_xlabel("Step", fontsize='large')
ax.set_ylabel("Loss", fontsize='large')
ax.legend()
plt.tight_layout()
fig.savefig(f"results/figures/{trainer_name}_loss.eps", format="eps")
