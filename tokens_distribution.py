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
    dataset = load_dataset(f"./datasets/{dataset_name}", "2_classes")

    proposals = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])["premise"]
    arguments = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])["hypothesis"]

    prop_tokens_distribution, prop_lowest_five_p, prop_highest_five_p = get_tokens_distribution(proposals)
    arg_tokens_distribution, arg_lowest_five_p, arg_highest_five_p = get_tokens_distribution(arguments)

    binwidth = 5
    textstyle = {'color': 'red', 'weight': 'heavy', 'size': 12}
    boxstyle = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}

    fig1, ax1 = plt.subplots()
    ax1.hist(arg_tokens_distribution, bins=range(0, max(arg_tokens_distribution) + binwidth, binwidth))
    ax1.axvline(arg_lowest_five_p, linestyle="dashed", color="r")
    ax1.axvline(arg_highest_five_p, linestyle="dashed", color="r")
    ax1.text(arg_lowest_five_p + 10, ax1.get_ylim()[1] / 2, int(arg_lowest_five_p), textstyle, bbox=boxstyle)
    ax1.text(arg_highest_five_p + 10, ax1.get_ylim()[1] / 2, int(arg_highest_five_p), textstyle, bbox=boxstyle)
    ax1.set_xlabel("Number of tokens", fontsize='large')
    ax1.set_ylabel("Number of texts", fontsize='large')
    ax1.set_title(f"With {dataset_name} dataset: arguments")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.hist(prop_tokens_distribution, bins=range(0, max(prop_tokens_distribution) + binwidth, binwidth))
    ax2.axvline(prop_lowest_five_p, linestyle="dashed", color="r")
    ax2.axvline(prop_highest_five_p, linestyle="dashed", color="r")
    ax2.text(prop_lowest_five_p + 10, ax2.get_ylim()[1] / 2, int(prop_lowest_five_p), textstyle, bbox=boxstyle)
    ax2.text(prop_highest_five_p + 10, ax2.get_ylim()[1] / 2, int(prop_highest_five_p), textstyle, bbox=boxstyle)
    ax2.set_xlabel("Number of tokens", fontsize='large')
    ax2.set_ylabel("Number of texts", fontsize='large')
    ax2.set_title(f"With {dataset_name} dataset: proposals")
    plt.tight_layout()
    plt.show()
