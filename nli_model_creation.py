import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from datasets import load_dataset, load_metric

xnli_datasets = load_dataset("./datasets/xnli_fr")

model_checkpoint = "camembert-base"
batch_size = 2

model_name = model_checkpoint.split("/")[-1]

label_list = xnli_datasets["train"].features["label"].names

config = AutoConfig.from_pretrained(model_checkpoint)

config.id2label = {idx: label for (idx, label) in enumerate(label_list)}
config.label2id = {label: idx for (idx, label) in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512, use_fast=True)

encoded_dataset = xnli_datasets.map(lambda examples: tokenizer(examples["premise"], examples["hypothesis"], max_length=512, truncation=True), batched=True)

num_labels = len(label_list)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

metric = load_metric('glue', "mnli")
metric_name = "accuracy"

model.config.name_or_path = f"waboucay/{model_name}-finetuned-xnli_fr"


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


args = TrainingArguments(
    f"{model_name}-finetuned-xnli_fr",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train(resume_from_checkpoint=True)
print("With validation set:")
print(trainer.evaluate())
print("With test set:")
print(trainer.evaluate(eval_dataset=encoded_dataset["test"]))

trainer.save_model(f"{model_name}-finetuned-nli-xnli_fr")
