# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{11403/perceo/v1,
    title = {PERCEO : un Projet d'Etiqueteur Robuste pour l'Ecrit et pour l'Oral},
    author = {ATILF and INIST and LIPN},
    url = {https://hdl.handle.net/11403/perceo/v1},
    note = {{ORTOLANG} ({Open} {Resources} {and} {TOols} {for} {LANGuage}) \textendash www.ortolang.fr},
    copyright = {Licence Creative Commons Attribution - Pas du2019Utilisation Commerciale - Partage dans les Mêmes Conditions 2.0 Générique},
    year = {2012}
}
"""

_DESCRIPTION = """\
"""

_URL = "https://repository.ortolang.fr/api/content/perceo/v1/perceo_oral.zip"
_CORPUS_FILE = "perceo_oral/corpus_perceo_oral.txt"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"
_TEST_FILE = "test.txt"

replace_tags = {
        "ADJ:trc": "ADJ",
        "AUX:cond": "VER:cond",
        "AUX:futu": "VER:futu",
        "AUX:impe": "VER:impe",
        "AUX:impf": "VER:impf",
        "AUX:infi": "VER:infi",
        "AUX:pper": "VER:pper",
        "AUX:ppre": "VER:ppre",
        "AUX:pres": "VER:pres",
        "AUX:simp": "VER:simp",
        "AUX:subi": "VER:subi",
        "AUX:subp": "VER:subp",
        "LOC": "SENT",
        "NAM:trc": "NAM",
        "NOM:trc": "NOM",
        "PRO:clo": "PRO",
        "PRO:cls": "PRO",
        "PRO:clsi": "PRO",
        "PRO:int": "PRO",
        "PRO:ton": "PRO"
}


class PerceoConfig(datasets.BuilderConfig):
    """BuilderConfig for Perceo"""

    def __init__(self, **kwargs):
        """BuilderConfig for Perceo.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PerceoConfig, self).__init__(**kwargs)


class Perceo(datasets.GeneratorBasedBuilder):
    """Perceo dataset."""

    BUILDER_CONFIGS = [
        PerceoConfig(name="Perceo", version=datasets.Version("1.0.0"), description="Perceo dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "ABR",
                                "ADJ",
                                "ADV",
                                "DET:art",
                                "DET:def",
                                "DET:dem",
                                "DET:ind",
                                "DET:int",
                                "DET:par",
                                "DET:pos",
                                "DET:pre",
                                "EPE",
                                "ETR",
                                "FNO",
                                "INT",
                                "KON",
                                "MLT",
                                "NAM",
                                "NAM:sig",
                                "NOM",
                                "NOM:sig",
                                "NUM",
                                "PRO",
                                "PRO:dem",
                                "PRO:ind",
                                "PRO:per",
                                "PRO:pos",
                                "PRO:rel",
                                "PRP",
                                "PRP:det",
                                "PRT:int",
                                "PUN",
                                "PUN:cit",
                                "SENT",
                                "SYM",
                                "TRC",
                                "VER",
                                "VER:cond",
                                "VER:futu",
                                "VER:impe",
                                "VER:impf",
                                "VER:infi",
                                "VER:pper",
                                "VER:ppre",
                                "VER:pres",
                                "VER:simp",
                                "VER:subi",
                                "VER:subp",
                                "VER:trc"
                            ]
                        )
                    ),
                    "lemmas": datasets.Sequence(datasets.Value("string"))
                }
            ),
            supervised_keys=None,
            homepage="https://hdl.handle.net/11403/perceo",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)

        # Splitting the corpus into three files
        corpus_path = os.path.join(downloaded_file, _CORPUS_FILE)
        training_path = os.path.join(downloaded_file, _TRAINING_FILE)
        dev_path = os.path.join(downloaded_file, _DEV_FILE)
        test_path = os.path.join(downloaded_file, _TEST_FILE)

        with open(corpus_path, "r") as corpus_file, open(training_path, "w") as training_file, open(dev_path, "w") as dev_file, open(test_path, "w") as test_file:
            corpus_size = sum(1 if "\tLOC\t" in line else 0 for line in corpus_file)

            corpus_file.seek(0)
            file_to_write = training_file

            sentence_idx = 0

            for line in corpus_file:
                file_to_write.write(line)

                if "\tLOC\t" in line:
                    sentence_idx += 1

                if sentence_idx > corpus_size / 3 and file_to_write == training_file and "\tLOC\t" in line:
                    file_to_write = dev_file
                elif sentence_idx > 2 * corpus_size / 3 and file_to_write == dev_file and "\tLOC\t" in line:
                    file_to_write = test_file

        data_files = {
            "train": training_path,
            "dev": dev_path,
            "test": test_path,
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            pos_tags = []
            lemmas = []

            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n" or "\tLOC\t" in line:
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                            "lemmas": lemmas
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                        lemmas = []
                else:
                    # Perceo tokens are tab separated
                    splits = line.strip().split("\t")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1] if splits[1] not in replace_tags.keys() else replace_tags[splits[1]])
                    lemmas.append(splits[2])
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "pos_tags": pos_tags,
                "lemmas": lemmas
            }
