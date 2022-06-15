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

from utils.functions import get_original_proposal_repnum

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_CITATION = """\
@misc{11403/perceo/v1,
    title = {Données de la consultation sur le projet de loi pour une République Numérique},
    author = {Gouvernement Français},
    url = {https://www.data.gouv.fr/fr/datasets/consultation-sur-le-projet-de-loi-republique-numerique/},
    copyright = {Licence Ouverte/Open License Version 2.0},
    year = {2015}
}
"""

_URL = "https://www.data.gouv.fr/fr/datasets/r/891bca8a-d9c1-4250-bfb2-3d13bf595813"
_TRAINING_FILE = "train.csv"
_DEV_FILE = "valid.csv"
_TEST_FILE = "test.csv"

class RepNumNliConfig(datasets.BuilderConfig):
    """BuilderConfig for RepNumNli."""

    def __init__(self, **kwargs):
        """BuilderConfig for RepNumNli.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RepNumNliConfig, self).__init__(**kwargs)


class RepNumNli(datasets.GeneratorBasedBuilder):
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    VERSION = datasets.Version("1.0.0", "")
    BUILDER_CONFIG_CLASS = RepNumNliConfig
    BUILDER_CONFIGS = [
        RepNumNliConfig(
            name="3_classes",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of RepNum consultation NLI (weakly labeled)",
        ),
        RepNumNliConfig(
            name="2_classes",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of RepNum consultation NLI (weakly labeled), only 2 classes: Non-contradictory (0) or Contradictory (1)",
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
            homepage="https://republique-numerique.fr/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_file = dl_manager.download(_URL)

        training_path = os.path.join(os.path.dirname(dl_file), _TRAINING_FILE)
        eval_path = os.path.join(os.path.dirname(dl_file), _DEV_FILE)
        test_path = os.path.join(os.path.dirname(dl_file), _TEST_FILE)

        consultation_data = pd.read_csv(dl_file, parse_dates=["Création", "Modification"], index_col=0,
                                        dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str,
                                               "Lien": str})
        consultation_data["Lié.à.."] = consultation_data["Lié.à.."].fillna("Unknown")
        consultation_data["Type.de.profil"] = consultation_data["Type.de.profil"].fillna("Unknown")

        arguments = consultation_data.loc[consultation_data["Type.de.contenu"] == "Argument"]

        arguments["Initial_proposal"] = arguments.apply(lambda row: get_original_proposal_repnum(row, consultation_data)["Contenu"], axis=1)

        argument_for_label = "non-contradiction" if self.config.name == "2_classes" else "entailment"
        arguments["label"] = arguments["Catégorie"].apply(lambda category: argument_for_label if category == "Pour" else "contradiction")

        training_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])
        eval_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])
        test_dataset = pd.DataFrame(columns=['premise', 'hypothesis', 'label'])

        for i in range(arguments.shape[0]):
            row = arguments.iloc[i]

            formatted_row = pd.DataFrame({'premise': [row["Initial_proposal"]], 'hypothesis': [row["Contenu"]], 'label': [row["label"]]})

            if i % 10 < 8:
                training_dataset = pd.concat([training_dataset, formatted_row], ignore_index=True)
            elif i % 10 < 9:
                eval_dataset = pd.concat([eval_dataset, formatted_row], ignore_index=True)
            else:
                test_dataset = pd.concat([test_dataset, formatted_row], ignore_index=True)

        if self.config.name == "3_classes":
            proposals = consultation_data.loc[consultation_data["Type.de.contenu"] == "Proposition"]

            unrelated_arguments = consultation_data.loc[consultation_data["Type.de.contenu"] == "Argument"].sample(n=int(len(arguments.index) / 2), random_state=1234)
            unrelated_arguments["Initial_category"] = unrelated_arguments.apply(lambda row: get_original_proposal_repnum(row, consultation_data)["Catégorie"], axis=1)
            unrelated_arguments["Initial_proposal"] = unrelated_arguments.apply(lambda row: proposals.loc[proposals["Catégorie"] != row["Initial_category"]].sample(random_state=int(row["Identifiant"]))["Contenu"].iloc[0], axis=1)
            for i in range(len(unrelated_arguments.index)):
                row = unrelated_arguments.iloc[i]

                formatted_row = pd.DataFrame(
                    {'premise': [row["Initial_proposal"]], 'hypothesis': [row["Contenu"]], 'label': ["neutral"]})

                if i % 10 < 8:
                    training_dataset = pd.concat([training_dataset, formatted_row], ignore_index=True)
                elif i % 10 < 9:
                    eval_dataset = pd.concat([eval_dataset, formatted_row], ignore_index=True)
                else:
                    test_dataset = pd.concat([test_dataset, formatted_row], ignore_index=True)

            training_dataset = training_dataset.sample(frac=1).reset_index(drop=True)
            eval_dataset = eval_dataset.sample(frac=1).reset_index(drop=True)
            test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)

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
