import re

import pandas as pd
from datasets import DatasetDict, concatenate_datasets

from tokens_distribution import get_tokens_distribution


def remove_past_sentences(proposal_content, sentences_tokenizer, nlp_token_classifier):
    past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]

    sentences = sentences_tokenizer.tokenize(proposal_content)
    not_past_sentences = []

    for sentence in sentences:
        sentence_pos = nlp_token_classifier(sentence)
        if not any(token["entity"] in past_tense_tags for token in sentence_pos):
            not_past_sentences.append(sentence)

    return " ".join(not_past_sentences)


def predict_nli(premise, hypothesis, nli_tokenizer, nli_model) -> int:
    x = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=512, truncation=True)
    logits = nli_model(x)[0]
    probs = logits[:, ::].softmax(dim=1)
    return int(probs.detach().argmax() == 1)


def get_original_proposal_repnum(reply_contribution: pd.Series, previous_contributions: pd.DataFrame) -> pd.Series:
    related_to = reply_contribution["Lié.à.."]

    original_post_id = re.search('\d+', related_to).group()
    original_contribution_type = re.search('Proposition|Modification|Source|Argument', related_to).group()

    original_contribution = previous_contributions.loc[
        (previous_contributions["Identifiant"] == original_post_id) &
        (previous_contributions["Type.de.contenu"] == original_contribution_type)].iloc[0]

    return original_contribution


def get_original_proposal_rua(reply_contribution: pd.Series, previous_contributions: pd.DataFrame) -> pd.Series:
    original_post_id = reply_contribution["contributions_arguments_related_id"]

    original_contribution = previous_contributions.loc[previous_contributions["contributions_id"] == original_post_id].iloc[0]
    if original_contribution["contributions_trashed"] == 1:
        original_contribution["contributions_bodyText"] = ""

    return original_contribution


def remove_outliers_from_datasets(dataset_dict: DatasetDict) -> DatasetDict:
    result_datasets = DatasetDict({"train": [], "validation": [], "test": []})

    nb_tokens, min_tokens, max_tokens = get_tokens_distribution(concatenate_datasets([dataset_dict["train"], dataset_dict["validation"], dataset_dict["test"]])["hypothesis"])
    result_datasets["test"] = dataset_dict["test"].filter(lambda row, idx: min_tokens <= nb_tokens[idx + len(dataset_dict["train"]) + len(dataset_dict["validation"])] <= max_tokens, with_indices=True)
    result_datasets["validation"] = dataset_dict["test"].filter(lambda row, idx: min_tokens <= nb_tokens[idx + len(dataset_dict["train"])] <= max_tokens, with_indices=True)
    result_datasets["train"] = dataset_dict["train"].filter(lambda row, idx: min_tokens <= nb_tokens[idx] <= max_tokens, with_indices=True)

    return result_datasets
