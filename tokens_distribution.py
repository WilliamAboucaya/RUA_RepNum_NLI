from collections.abc import Iterable
from typing import Tuple

from datasets import load_dataset, concatenate_datasets
from nltk import word_tokenize

import matplotlib.pyplot as plt
import numpy as np


def get_tokens_distribution(texts: Iterable[str], quantile_1: float = 0.05, quantile_2: float = 0.95) -> Tuple[list[int], int, int]:
    nb_tokens = []
    for text in texts:
        nb_tokens.append(len(word_tokenize(text, language='french')))

    lowest_five_percent = np.quantile(nb_tokens, quantile_1)
    highest_five_percent = np.quantile(nb_tokens, quantile_2)
    return nb_tokens, lowest_five_percent, highest_five_percent


if __name__ == "__main__":
    dataset_name = "rua_nli"
    dataset = load_dataset(f"./datasets/{dataset_name}")

    arguments = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])["hypothesis"]

    tokens_distribution, lowest_five_p, highest_five_p = get_tokens_distribution(arguments)

    binwidth = 5
    textstyle = {'color': 'red', 'weight': 'heavy', 'size': 12}
    boxstyle = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

    fig, ax = plt.subplots()
    ax.hist(tokens_distribution, bins=range(0, max(tokens_distribution) + binwidth, binwidth))
    ax.axvline(lowest_five_p, linestyle="dashed", color="r")
    ax.axvline(highest_five_p, linestyle="dashed", color="r")
    ax.text(lowest_five_p + 10, ax.get_ylim()[1] / 2, int(lowest_five_p), textstyle, bbox=boxstyle)
    ax.text(highest_five_p + 10, ax.get_ylim()[1] / 2, int(highest_five_p), textstyle, bbox=boxstyle)
    ax.set_xlabel("Number of tokens", fontsize='large')
    ax.set_ylabel("Number of texts", fontsize='large')
    plt.tight_layout()
    plt.show()
