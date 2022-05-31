import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric, DatasetDict, concatenate_datasets
from pprint import pprint

from utils.functions import remove_outliers_from_datasets

assert torch.cuda.is_available()

# xnli_datasets = load_dataset("./datasets/xnli_fr", "2_classes")
# repnum_datasets = remove_outliers_from_datasets(load_dataset("./datasets/repnum_nli"))
rua_datasets = remove_outliers_from_datasets(load_dataset("./datasets/rua_nli"))

# train_dataset = concatenate_datasets([xnli_datasets["train"], repnum_datasets["train"], rua_datasets["train"]])
# eval_dataset = concatenate_datasets([xnli_datasets["validation"], repnum_datasets["validation"], rua_datasets["validation"]])
# test_dataset = concatenate_datasets([xnli_datasets["test"], repnum_datasets["test"], rua_datasets["test"]])

# train_dataset = concatenate_datasets([repnum_datasets["train"], rua_datasets["train"]])
# eval_dataset = concatenate_datasets([repnum_datasets["validation"], rua_datasets["validation"]])
# test_dataset = concatenate_datasets([repnum_datasets["test"], rua_datasets["test"]])

# train_dataset = xnli_datasets["train"]
# eval_dataset = xnli_datasets["validation"]
# test_dataset = xnli_datasets["test"]

train_dataset = rua_datasets["train"]
eval_dataset = rua_datasets["validation"]
test_dataset = rua_datasets["test"]

# train_dataset = repnum_datasets["train"]
# eval_dataset = repnum_datasets["validation"]
# test_dataset = repnum_datasets["test"]

nli_datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset}).shuffle(seed=1234)

model_checkpoint = "camembert-base"
batch_size = 8

model_name = model_checkpoint.split("/")[-1]

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

model.config.name_or_path = f"waboucay/{model_name}-finetuned-rua_wl"


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "f1_micro": metric.compute(predictions=predictions, references=labels, average="micro")["f1"],
        "f1_macro": metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    }


args = TrainingArguments(
    f"{model_name}-finetuned-rua_wl",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print("With validation set:")
pprint(trainer.evaluate())
print("With test set:")
pprint(trainer.evaluate(eval_dataset=encoded_dataset["test"]))

trainer.save_model(f"{model_name}-finetuned-nli-rua_wl")
