# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""RUA NLI without past sentences"""


import csv
import os

import datasets
import nltk
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from utils.functions import get_original_proposal_rua, remove_past_sentences

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_CITATION = """\
@misc{rua_nli,
    title = {Données de la consultation sur le Revenu Universel d'activité},
    author = {Gouvernement Français},
    url = {https://www.data.gouv.fr/fr/datasets/consultation-vers-un-revenu-universel-dactivite-1/},
    year = {2019}
}
"""

_URLS = {
    "fonctionnement": "https://github.com/WilliamAboucaya/rua_opendata_corrected/raw/main/rua-fonctionnement.csv",
    "publics": "https://github.com/WilliamAboucaya/rua_opendata_corrected/raw/main/rua-publics.csv",
    "principes": "https://github.com/WilliamAboucaya/rua_opendata_corrected/raw/main/rua-principes.csv"
}
_TRAINING_FILE = "train.csv"
_DEV_FILE = "valid.csv"
_TEST_FILE = "test.csv"

class RuaNliNoPastConfig(datasets.BuilderConfig):
    """BuilderConfig for RuaNli."""

    def __init__(self, **kwargs):
        """BuilderConfig for RuaNli.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RuaNliNoPastConfig, self).__init__(**kwargs)


class RuaNli(datasets.GeneratorBasedBuilder):
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    VERSION = datasets.Version("1.0.0", "")
    BUILDER_CONFIG_CLASS = RuaNliNoPastConfig
    BUILDER_CONFIGS = [
        RuaNliNoPastConfig(
            name="3_classes",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of RUA consultation NLI (weakly labeled)",
        ),
        RuaNliNoPastConfig(
            name="2_classes",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of RUA consultation NLI (weakly labeled), only 2 classes: Non-contradictory (0) or Contradictory (1)",
        )
    ]

    def _info(self):
        if self.config.name == "2_classes":
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["non-contradiction", "contradiction"]),
                }
            )
        else:
            features = datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["entailment", "contradiction", "neutral"]),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://consultation-rua.gouv.fr/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_files = dl_manager.download(_URLS)

        training_path = os.path.join(os.path.dirname(dl_files["fonctionnement"]), _TRAINING_FILE)
        eval_path = os.path.join(os.path.dirname(dl_files["fonctionnement"]), _DEV_FILE)
        test_path = os.path.join(os.path.dirname(dl_files["fonctionnement"]), _TEST_FILE)

        training_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])
        eval_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])
        test_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])

        dataframes = []
        for dl_file in dl_files.values():
            dataframes.append(pd.read_csv(dl_file, encoding="utf8", engine='python', sep=';', quotechar='"'))

        consultation_data = pd.concat(dataframes, ignore_index=True)
        consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")

        arguments = consultation_data.loc[consultation_data["type"] == "argument"]

        arguments = arguments[arguments["contributions_arguments_trashed"] != 1]

        arguments["initial_proposal"] = arguments.apply(lambda row: get_original_proposal_rua(row, consultation_data)["contributions_bodyText"], axis=1)
        arguments = arguments[(arguments["initial_proposal"] != "") &
                              (arguments["contributions_arguments_body"] != "")]

        argument_for_label = "non-contradiction" if self.config.name == "2_classes" else "entailment"
        arguments["label"] = arguments["contributions_arguments_type"].apply(lambda category: argument_for_label if category == "FOR" else "contradiction")

        for i in range(len(arguments.index)):
            row = arguments.iloc[i]

            formatted_row = pd.DataFrame({'premise': [row["initial_proposal"]], 'hypothesis': [row["contributions_arguments_body"]], 'label': [row["label"]]})

            if i % 10 < 8:
                training_dataset = pd.concat([training_dataset, formatted_row], ignore_index=True)
            elif i % 10 < 9:
                eval_dataset = pd.concat([eval_dataset, formatted_row], ignore_index=True)
            else:
                test_dataset = pd.concat([test_dataset, formatted_row], ignore_index=True)

        if self.config.name == "3_classes":
            proposals = consultation_data.loc[consultation_data["type"] == "opinion"]

            unrelated_arguments = consultation_data.loc[consultation_data["type"] == "argument"].sample(n=int(len(arguments.index) / 2), random_state=1234)
            unrelated_arguments["initial_category"] = unrelated_arguments.apply(lambda row: get_original_proposal_rua(row, consultation_data)["contributions_consultation_title"], axis=1)
            unrelated_arguments["initial_proposal"] = unrelated_arguments.apply(lambda row: proposals.loc[proposals["contributions_consultation_title"] != row["initial_category"]].sample(random_state=int(row.name))["contributions_bodyText"].iloc[0], axis=1)
            for i in range(len(unrelated_arguments.index)):
                row = unrelated_arguments.iloc[i]

                formatted_row = pd.DataFrame(
                    {'premise': [row["initial_proposal"]], 'hypothesis': [row["contributions_arguments_body"]], 'label': ["neutral"]})

                if i % 10 < 8:
                    training_dataset = pd.concat([training_dataset, formatted_row], ignore_index=True)
                elif i % 10 < 9:
                    eval_dataset = pd.concat([eval_dataset, formatted_row], ignore_index=True)
                else:
                    test_dataset = pd.concat([test_dataset, formatted_row], ignore_index=True)

            training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
            eval_dataset = eval_dataset.sample(frac=1).reset_index(drop=True)
            test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)

        sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
        pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
        pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
        nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)
        test_dataset["premise"] = test_dataset["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
        test_dataset["hypothesis"] = test_dataset["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))

        training_dataset.to_csv(training_path, encoding="utf-8")
        eval_dataset.to_csv(eval_path, encoding="utf-8")
        test_dataset.to_csv(test_path, encoding="utf-8")

        data_files = {
            "train": training_path,
            "dev": eval_path,
            "test": test_path,
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""

        with open(filepath, encoding="utf-8") as f:
            guid = 0

            reader = csv.DictReader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                yield guid, {
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"]
                }
                guid += 1
