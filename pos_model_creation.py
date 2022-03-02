from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, AutoConfig
from datasets import load_dataset, load_metric

import numpy as np

label_all_tokens = True
model_checkpoint = "gilf/french-camembert-postag-model"
batch_size = 2

model_name = model_checkpoint.split("/")[-1]

perceo_datasets = load_dataset('./datasets/perceo')
label_list = perceo_datasets["train"].features["pos_tags"].feature.names

config = AutoConfig.from_pretrained(model_checkpoint)

config.name_or_path = f"waboucay/{model_name}-finetuned-perceo"
config.id2label = {idx: label for (idx, label) in enumerate(label_list)}
config.label2id = {label: idx for (idx, label) in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=512)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], max_length=512, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
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
    return tokenized_inputs


def compute_metrics(p):
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

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision_micro": results["overall_precision_micro"],
        "recall_micro": results["overall_recall_micro"],
        "f1_micro": results["overall_f1_micro"],
        "f1_macro": results["overall_f1_macro"],
        "f1_VER:impf": results["VER:impf"]["f1"],
        "f1_VER:simp": results["VER:simp"]["f1"],
        "f1_VER:subi": results["VER:subi"]["f1"],
        "accuracy": results["overall_accuracy"],
    }
    # accuracy_results = accuracy_metric.compute(predictions=true_predictions, references=true_labels)
    # f1_micro_results = f1_metric.compute(predictions=true_predictions, references=true_labels, average="micro")
    # f1_macro_results = f1_metric.compute(predictions=true_predictions, references=true_labels, average="macro")
    # f1_impf_results = f1_metric.compute(predictions=true_predictions, references=true_labels, average="binary", pos_label=40)
    # f1_simp_results = f1_metric.compute(predictions=true_predictions, references=true_labels, average="binary", pos_label=45)
    # f1_subi_results = f1_metric.compute(predictions=true_predictions, references=true_labels, average="binary", pos_label=46)
    # precision_results = precision_metric.compute(predictions=true_predictions, references=true_labels)
    # recall_results = recall_metric.compute(predictions=true_predictions, references=true_labels)
    # return {
    #     "precision": precision_results["precision"],
    #     "recall": recall_results["recall"],
    #     "f1_micro": f1_micro_results["f1"],
    #     "f1_macro": f1_macro_results["f1"],
    #     "f1_VER:impf": f1_impf_results["f1"],
    #     "f1_VER:simp": f1_simp_results["f1"],
    #     "f1_VER:subi": f1_subi_results["f1"],
    #     "accuracy": accuracy_results["accuracy"],
    # }


perceo_datasets = load_dataset('./datasets/perceo')
label_list = perceo_datasets["train"].features["pos_tags"].feature.names

tokenized_datasets = perceo_datasets.map(tokenize_and_align_labels, batched=True)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config, ignore_mismatched_sizes=True)

args = TrainingArguments(
    f"{model_name}-finetuned-pos",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("./metrics/seqeval_exhaustive")
# accuracy_metric = load_metric("accuracy")
# f1_metric = load_metric("f1")
# precision_metric = load_metric("precision")
# recall_metric = load_metric("recall")

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
print("With validation set:")
print(trainer.evaluate())
print("With test set:")
print(trainer.evaluate(eval_dataset=tokenized_datasets["validation"]))

trainer.save_model(f"{model_name}-finetuned-perceo")
