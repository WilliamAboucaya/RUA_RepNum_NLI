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
"""XNLI: The Cross-Lingual NLI Corpus."""


import csv
import os

import datasets


_CITATION = """\
@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}"""

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_TRAIN_DATA_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip"
_TESTVAL_DATA_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"


class XnliFrConfig(datasets.BuilderConfig):
    """BuilderConfig for XNLI."""

    def __init__(self, **kwargs):
        """BuilderConfig for XNLI.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(XnliFrConfig, self).__init__(**kwargs)


class XnliFr(datasets.GeneratorBasedBuilder):
    """XNLI: The Cross-Lingual NLI Corpus. Version 1.0."""

    VERSION = datasets.Version("1.1.0", "")
    BUILDER_CONFIG_CLASS = XnliFrConfig
    BUILDER_CONFIGS = [
        XnliFrConfig(
            name="3_classes",
            version=datasets.Version("1.1.0", ""),
            description="Plain text import of XNLI for the fr language",
        ),
        XnliFrConfig(
            name="2_classes",
            version=datasets.Version("1.1.0", ""),
            description="Plain text import of XNLI for the fr language, only 2 classes: Non-contradictory (0) or Contradictory (1)",
        ),
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
            homepage="https://www.nyu.edu/projects/bowman/xnli/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dirs = dl_manager.download_and_extract(
            {
                "train_data": _TRAIN_DATA_URL,
                "testval_data": _TESTVAL_DATA_URL,
            }
        )
        train_dir = os.path.join(dl_dirs["train_data"], "XNLI-MT-1.0", "multinli")
        testval_dir = os.path.join(dl_dirs["testval_data"], "XNLI-1.0")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": [os.path.join(train_dir, f"multinli.train.fr.tsv")],
                    "data_format": "XNLI-MT",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": [os.path.join(testval_dir, "xnli.test.tsv")], "data_format": "XNLI"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": [os.path.join(testval_dir, "xnli.dev.tsv")], "data_format": "XNLI"},
            ),
        ]

    def _generate_examples(self, data_format, filepaths):
        """This function returns the examples in the raw (text) form."""

        if self.config.name == "2_classes":
            if data_format == "XNLI-MT":
                for file_idx, filepath in enumerate(filepaths):
                    file = open(filepath, encoding="utf-8")
                    reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row_idx, row in enumerate(reader):
                        key = str(file_idx) + "_" + str(row_idx)
                        yield key, {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": "contradiction" if row["label"] == "contradictory" else "non-contradiction",
                        }
            else:
                for filepath in filepaths:
                    with open(filepath, encoding="utf-8") as f:
                        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                        for row in reader:
                            if row["language"] == "fr":
                                yield row["pairID"], {
                                    "premise": row["sentence1"],
                                    "hypothesis": row["sentence2"],
                                    "label": row["gold_label"] if row["gold_label"] == "contradiction" else "non-contradiction",
                                }
        else:
            if data_format == "XNLI-MT":
                for file_idx, filepath in enumerate(filepaths):
                    file = open(filepath, encoding="utf-8")
                    reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
                    for row_idx, row in enumerate(reader):
                        key = str(file_idx) + "_" + str(row_idx)
                        yield key, {
                            "premise": row["premise"],
                            "hypothesis": row["hypo"],
                            "label": "contradiction" if row["label"] == "contradictory" else row["label"],
                        }
            else:
                for filepath in filepaths:
                    with open(filepath, encoding="utf-8") as f:
                        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                        for row in reader:
                            if row["language"] == "fr":
                                yield row["pairID"], {
                                    "premise": row["sentence1"],
                                    "hypothesis": row["sentence2"],
                                    "label": row["gold_label"]
                                }
