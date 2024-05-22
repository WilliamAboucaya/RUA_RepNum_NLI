import re
import math
from typing import Tuple

import nltk
import numpy as np
from datasets import Metric
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from scipy.optimize import dual_annealing
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from tokens_distribution import get_tokens_distribution


import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    return int(probs.detach().argmax())


def predict_nli_batch(batch, nli_tokenizer, nli_model, with_scores=False):
    x = nli_tokenizer(batch, return_tensors='pt', max_length=512, padding=True, truncation=True).to(device)
    logits = nli_model(x['input_ids'], attention_mask=x['attention_mask']).logits
    if with_scores:
        return logits.argmax(dim=1), logits.softmax(dim=1)
    return logits.argmax(dim=1)


def apply_model_sentencewise(row, sentences_tokenizer, nli_tokenizer, nli_model):
    premise_sentences = sentences_tokenizer.tokenize(row["premise"])
    hypothesis_sentences = sentences_tokenizer.tokenize(row["hypothesis"])

    nb_entailed_pairs = 0
    nb_contradictory_pairs = 0
    nb_neutral_pairs = 0

    for i in range(len(premise_sentences)):
        for j in range(len(hypothesis_sentences)):
            predicted_label = predict_nli(premise_sentences[i], hypothesis_sentences[j], nli_tokenizer, nli_model)

            if predicted_label == 0:
                nb_entailed_pairs +=1
            elif predicted_label == 1:
                nb_contradictory_pairs += 1
            else:
                nb_neutral_pairs += 1

    try:
        share_entailed_pairs = nb_entailed_pairs / (len(premise_sentences) * len(hypothesis_sentences))
    except ZeroDivisionError:
        share_entailed_pairs = 0
    try:
        share_contradictory_pairs = nb_contradictory_pairs / (len(premise_sentences) * len(hypothesis_sentences))
    except ZeroDivisionError:
        share_contradictory_pairs = 0
    try:
        share_neutral_pairs = nb_neutral_pairs / (len(premise_sentences) * len(hypothesis_sentences))
    except ZeroDivisionError:
        share_neutral_pairs = 0

    return nb_entailed_pairs, share_entailed_pairs, nb_contradictory_pairs, share_contradictory_pairs, nb_neutral_pairs, share_neutral_pairs


def apply_model_sentencewise_batch(proposals, sentences_tokenizer, nli_tokenizer, nli_model, batch_size):
    proposals["nb_entailed_pairs"] = np.nan
    proposals["share_entailed_pairs"] = np.nan
    proposals["nb_contradictory_pairs"] = np.nan
    proposals["share_contradictory_pairs"] = np.nan
    proposals["nb_neutral_pairs"] = np.nan
    proposals["share_neutral_pairs"] = np.nan
    positions_tuples = []
    sentences = []
    predictions = []
    for i in range(len(proposals)):
        positions_tuples.append(len(sentences))
        premise_sentences = sentences_tokenizer.tokenize(proposals.at[i, "premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(proposals.at[i, "hypothesis"])
        for j in range(len(premise_sentences)):
            for k in range(len(hypothesis_sentences)):
                sentences.append((premise_sentences[j], hypothesis_sentences[k]))

    print("Sentences pair number = " + str(len(sentences)))
    nb_batches = int(math.ceil(len(sentences) / batch_size))
    for i in range(nb_batches):
        start_poz = i * batch_size
        stop_poz = min(start_poz + batch_size, len(sentences))
        batch = []
        for j in range(start_poz, stop_poz):
            batch.append(sentences[j])

        predicted_labels = predict_nli_batch(batch, nli_tokenizer, nli_model)
        for j in range(start_poz, stop_poz):
            predictions.append(predicted_labels[j - start_poz].item())

    for i in range(len(positions_tuples)):
        pos_start = positions_tuples[i]
        pos_end = len(sentences)
        if i < len(positions_tuples) - 1:
            pos_end = positions_tuples[i + 1]
        nb_entailed_pairs = 0
        nb_contradictory_pairs = 0
        nb_neutral_pairs = 0
        for j in range(pos_start, pos_end):
            predicted_label = predictions[j]
            if predicted_label == 0:
                nb_entailed_pairs += 1
            elif predicted_label == 1:
                nb_contradictory_pairs += 1
            else:
                nb_neutral_pairs += 1

        try:
            share_entailed_pairs = nb_entailed_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_entailed_pairs = 0
        try:
            share_contradictory_pairs = nb_contradictory_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_contradictory_pairs = 0
        try:
            share_neutral_pairs = nb_neutral_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_neutral_pairs = 0
        # nb_entailed_pairs, share_entailed_pairs, nb_contradictory_pairs, share_contradictory_pairs, nb_neutral_pairs, share_neutral_pairs

        proposals.at[i, "nb_entailed_pairs"] = nb_entailed_pairs
        proposals.at[i, "share_entailed_pairs"] = share_entailed_pairs
        proposals.at[i, "nb_contradictory_pairs"] = nb_contradictory_pairs
        proposals.at[i, "share_contradictory_pairs"] = share_contradictory_pairs
        proposals.at[i, "nb_neutral_pairs"] = nb_neutral_pairs
        proposals.at[i, "share_neutral_pairs"] = share_neutral_pairs



def apply_model_sentencecouple(row, sentences_tokenizer, nli_tokenizer, nli_model):
    premise_sentences = sentences_tokenizer.tokenize(row["premise"])
    hypothesis_sentences = sentences_tokenizer.tokenize(row["hypothesis"])

    nb_entailed_pairs = 0
    nb_contradictory_pairs = 0
    nb_neutral_pairs = 0

    for i in range(1, len(premise_sentences)):
        for j in range(1, len(hypothesis_sentences)):
            predicted_label = predict_nli(" ".join(premise_sentences[i - 1:i + 1]),
                                          " ".join(hypothesis_sentences[j - 1:j + 1]), nli_tokenizer, nli_model)

            if predicted_label == 0:
                nb_entailed_pairs +=1
            elif predicted_label == 1:
                nb_contradictory_pairs += 1
            else:
                nb_neutral_pairs += 1

    try:
        share_entailed_pairs = nb_entailed_pairs / ((len(premise_sentences) - 1) * (len(hypothesis_sentences) - 1))
    except ZeroDivisionError:
        share_entailed_pairs = 0
    try:
        share_contradictory_pairs = nb_contradictory_pairs / ((len(premise_sentences) - 1) * (len(hypothesis_sentences) - 1))
    except ZeroDivisionError:
        share_contradictory_pairs = 0
    try:
        share_neutral_pairs = nb_neutral_pairs / ((len(premise_sentences) - 1) * (len(hypothesis_sentences) - 1))
    except ZeroDivisionError:
        share_neutral_pairs = 0

    return nb_entailed_pairs, share_entailed_pairs, nb_contradictory_pairs, share_contradictory_pairs, nb_neutral_pairs, share_neutral_pairs


def apply_model_sentencecouple_batch(proposals, sentences_tokenizer, nli_tokenizer, nli_model, batch_size):
    proposals["nb_entailed_pairs"] = np.nan
    proposals["share_entailed_pairs"] = np.nan
    proposals["nb_contradictory_pairs"] = np.nan
    proposals["share_contradictory_pairs"] = np.nan
    proposals["nb_neutral_pairs"] = np.nan
    proposals["share_neutral_pairs"] = np.nan
    positions_tuples = []
    sentences = []
    predictions = []
    for i in range(len(proposals)):
        positions_tuples.append(len(sentences))
        premise_sentences = sentences_tokenizer.tokenize(proposals.at[i, "premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(proposals.at[i, "hypothesis"])
        for j in range(1, len(premise_sentences)):
            for k in range(1, len(hypothesis_sentences)):
                sentences.append(
                    (" ".join(premise_sentences[j - 1:j + 1]), " ".join(hypothesis_sentences[k - 1:k + 1])))

    print("Sentences pair number = " + str(len(sentences)))
    nb_batches = int(math.ceil(len(sentences) / batch_size))
    for i in range(nb_batches):
        start_poz = i * batch_size
        stop_poz = min(start_poz + batch_size, len(sentences))
        batch = []
        for j in range(start_poz, stop_poz):
            batch.append(sentences[j])

        predicted_labels = predict_nli_batch(batch, nli_tokenizer, nli_model)
        for j in range(start_poz, stop_poz):
            predictions.append(predicted_labels[j - start_poz].item())

    for i in range(len(positions_tuples)):
        pos_start = positions_tuples[i]
        pos_end = len(sentences)
        if i < len(positions_tuples) - 1:
            pos_end = positions_tuples[i + 1]
        nb_entailed_pairs = 0
        nb_contradictory_pairs = 0
        nb_neutral_pairs = 0
        for j in range(pos_start, pos_end):
            predicted_label = predictions[j]
            if predicted_label == 0:
                nb_entailed_pairs += 1
            elif predicted_label == 1:
                nb_contradictory_pairs += 1
            else:
                nb_neutral_pairs += 1

        try:
            share_entailed_pairs = nb_entailed_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_entailed_pairs = 0
        try:
            share_contradictory_pairs = nb_contradictory_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_contradictory_pairs = 0
        try:
            share_neutral_pairs = nb_neutral_pairs / (pos_end - pos_start)
        except ZeroDivisionError:
            share_neutral_pairs = 0
        # nb_entailed_pairs, share_entailed_pairs, nb_contradictory_pairs, share_contradictory_pairs, nb_neutral_pairs, share_neutral_pairs

        proposals.at[i, "nb_entailed_pairs"] = nb_entailed_pairs
        proposals.at[i, "share_entailed_pairs"] = share_entailed_pairs
        proposals.at[i, "nb_contradictory_pairs"] = nb_contradictory_pairs
        proposals.at[i, "share_contradictory_pairs"] = share_contradictory_pairs
        proposals.at[i, "nb_neutral_pairs"] = nb_neutral_pairs
        proposals.at[i, "share_neutral_pairs"] = share_neutral_pairs



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

    nb_tokens, min_tokens, max_tokens = get_tokens_distribution(concatenate_datasets([dataset_dict["train"], dataset_dict["validation"], dataset_dict["test"]])["hypothesis"], quantile_2=1)
    result_datasets["test"] = dataset_dict["test"].filter(lambda row, idx: min_tokens <= nb_tokens[idx + len(dataset_dict["train"]) + len(dataset_dict["validation"])] <= max_tokens, with_indices=True)
    result_datasets["validation"] = dataset_dict["test"].filter(lambda row, idx: min_tokens <= nb_tokens[idx + len(dataset_dict["train"])] <= max_tokens, with_indices=True)
    result_datasets["train"] = dataset_dict["train"].filter(lambda row, idx: min_tokens <= nb_tokens[idx] <= max_tokens, with_indices=True)

    return result_datasets


def maximize_f1_score(contradiction_shares: pd.Series, entailment_shares: pd.Series, labels: list, metric: Metric) -> Tuple[float, float, float]:
    shares_df = pd.concat([contradiction_shares.to_frame(name="share_contradictory_pairs"), entailment_shares.to_frame(name="share_entailed_pairs")], axis=1)

    result = dual_annealing(lambda threshold: -metric.compute(predictions=shares_df.apply(lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"], threshold[0], threshold[1]), axis=1).tolist(),
                                                              references=labels, average="macro")["f1"], bounds=[[0, 1], [0, 1]])

    return result['x'][0], result['x'][1], -result['fun']


def define_label(contradiction_share: float, entailment_share: float, contradiction_threshold: float, entailment_threshold: float) -> int:
    if (contradiction_share >= contradiction_threshold) and (entailment_share >= entailment_threshold):
        return int((contradiction_share - contradiction_threshold) > (entailment_share - entailment_threshold))
    elif contradiction_share >= contradiction_threshold:
        return 1
    elif entailment_share >= entailment_threshold:
        return 0
    return 2


def generate_proposals_pairs_repnum(proposals: pd.DataFrame, no_past: bool = False):
    proposals["part"] = proposals.apply(lambda row: row["Catégorie"].split(" - ")[0], axis=1)
    proposals["full_proposal"] = proposals.apply(lambda row: row["Titre"] + ". " + row["Contenu"], axis=1)
    return generate_proposals_pairs(proposals, no_past)


def generate_proposals_pairs_rua(proposals: pd.DataFrame, no_past: bool = False):
    proposals["part"] = proposals.apply(lambda row: row["contributions_section_title"], axis=1)
    proposals["full_proposal"] = proposals.apply(lambda row: row["contributions_title"] + ". " + row["contributions_bodyText"], axis=1)
    return generate_proposals_pairs(proposals, no_past)


def generate_proposals_pairs(proposals: pd.DataFrame, no_past: bool = False):
    pos_model_path = "waboucay/french-camembert-postag-model-finetuned-perceo"  # TODO: USE pos_model_creation.py to re-create the POS tagger

    if no_past:
        sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
        pos_model = AutoModelForTokenClassification.from_pretrained(pos_model_path).to(device)
        pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_path)
        nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer, device=0)

        proposals["full_proposal"] = proposals["full_proposal"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))

    proposals_by_part = proposals.groupby("part")

    result_df = pd.DataFrame(columns=['premise', 'premise_idx', 'hypothesis', 'hypothesis_idx', 'part'])

    for name, df in proposals_by_part:
        for idx, premise in df["full_proposal"].iteritems():
            if premise == "":
                continue
            for idx2, hypothesis in df.loc[df.index > idx]["full_proposal"].iteritems():
                if hypothesis == "":
                    continue
                formatted_row = pd.DataFrame({'premise': [premise], 'premise_idx': [idx],
                                              'hypothesis': [hypothesis], 'hypothesis_idx': [idx2], 'part': [name]})
                result_df = pd.concat([result_df, formatted_row], ignore_index=True)

    return result_df
