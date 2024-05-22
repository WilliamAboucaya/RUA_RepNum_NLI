import pandas as pd
import torch

from datasets import Dataset
from evaluate import evaluator, load

if __name__ == "__main__":
    task_evaluator = evaluator("text-classification")
    model_name = "waboucay/camembert-large-finetuned-xnli_fr_3_classes-finetuned-rua_wl_3_classes"
    input_consultation_names = ["repnum", "repnum_with_titles", "rua", "rua_with_titles", "rua_with_titles_section"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"With model = {model_name}")

    for input_consultation_name in input_consultation_names:
        labeled_proposals_df = pd.read_csv(f"./consultation_data/nli_labeled_proposals_{input_consultation_name}.csv",
                                        encoding="utf8", engine='python', quoting=0, sep=';', dtype={"label": int})
        labeled_proposals_dst = Dataset.from_pandas(labeled_proposals_df)

        eval_results = task_evaluator.compute(
            model_or_pipeline=model_name,
            data=labeled_proposals_dst,
            label_mapping={
                "entailment": 0,
                "contradiction": 1,
                "neutral": 2
            },
            metric="./metrics/f1_macro",
            strategy="bootstrap",
            n_resamples=10000,
            input_column="premise",
            second_input_column="hypothesis",
            label_column="label",
            device=device
        )

        print(f'For {input_consultation_name:}')
        print(eval_results["f1"])
