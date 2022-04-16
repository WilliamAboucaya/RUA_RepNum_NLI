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
"""RepNum NLI"""


import csv
import os

import datasets
import pandas as pd

from utils import get_original_proposal_repnum, get_original_proposal_rua

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

class RuaNliConfig(datasets.BuilderConfig):
    """BuilderConfig for RuaNli."""

    def __init__(self, **kwargs):
        """BuilderConfig for RuaNli.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RuaNliConfig, self).__init__(**kwargs)


class RuaNli(datasets.GeneratorBasedBuilder):
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    VERSION = datasets.Version("1.0.0", "")
    BUILDER_CONFIG_CLASS = RuaNliConfig
    BUILDER_CONFIGS = [
        RuaNliConfig(
            name="rua_nli",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of RUA consultation NLI (weakly labeled)",
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["non-contradiction", "contradiction"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://republique-numerique.fr/",
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

        for dl_file in dl_files.values():

            consultation_data = pd.read_csv(dl_file, encoding="utf8", engine='python', sep=';', quotechar='"')
            consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")

            arguments = consultation_data.loc[consultation_data["type"] == "argument"]

            arguments = arguments[arguments["contributions_arguments_trashed"] != 1]

            arguments["initial_proposal"] = arguments.apply(lambda row: get_original_proposal_rua(row, consultation_data)["contributions_bodyText"], axis=1)
            arguments = arguments[(arguments["initial_proposal"] != "") &
                                  (arguments["contributions_arguments_body"] != "")]

            arguments["label"] = arguments["contributions_arguments_type"].apply(lambda category: "non-contradiction" if category == "FOR" else "contradiction")

            for i in range(arguments.shape[0]):
                row = arguments.iloc[i]

                formatted_row = pd.DataFrame({'premise': [row["initial_proposal"]], 'hypothesis': [row["contributions_arguments_body"]], 'label': [row["label"]]})

                if i % 10 < 8:
                    training_dataset = pd.concat([training_dataset, formatted_row], ignore_index=True)
                elif i % 10 < 9:
                    eval_dataset = pd.concat([eval_dataset, formatted_row], ignore_index=True)
                else:
                    test_dataset = pd.concat([test_dataset, formatted_row], ignore_index=True)

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
