import importlib
import sys

import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

if len(sys.argv) >= 5:
    dataset_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    model_revision = sys.argv[3]
    strategy_to_apply = sys.argv[4]
else:
    dataset_name = "rua_nli"
    model_checkpoint = "waboucay/camembert-large-finetuned-repnum_wl-rua_wl"
    model_revision = "main"
    strategy_to_apply = "withpast_sentencecouple_contradictionshare"

dataset = load_dataset(f"../datasets/{dataset_name}", "3_classes")

print(f'dataset_name = {dataset_name}')
print(f'model_checkpoint = {model_checkpoint}')
print(f'model_revision = {model_revision}')
print(f'strategy_to_apply = {strategy_to_apply}')

apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply}").apply_strategy

train_df = apply_strategy(dataset["train"].to_pandas(), model_checkpoint, model_revision)
print("Strategy applied on training set")
validation_df = apply_strategy(dataset["validation"].to_pandas(), model_checkpoint, model_revision)
print("Strategy applied on validation set")
test_df = apply_strategy(dataset["test"].to_pandas(), model_checkpoint, model_revision)
print("Strategy applied on test set")

classifier = LogisticRegression().fit(train_df[["share_contradictory_pairs", "nb_contradictory_pairs"]].to_numpy(), train_df["label"].to_numpy())
print("Model trained!")

print("On validation set:", classifier.score(validation_df[["share_contradictory_pairs", "nb_contradictory_pairs"]].to_numpy(), validation_df["label"].to_numpy()))
print("On test set:", classifier.score(test_df[["share_contradictory_pairs", "nb_contradictory_pairs"]].to_numpy(), test_df["label"].to_numpy()))

joblib.dump(classifier, f"../results/joblib_dumps/{dataset_name}/classifier_{model_checkpoint.split('/')[-1]}_{strategy_to_apply}.joblib")
