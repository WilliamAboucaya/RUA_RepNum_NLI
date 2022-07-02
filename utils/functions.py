import re
from typing import Tuple

from datasets import Metric
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from scipy.optimize import dual_annealing

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
    return int(probs.detach().argmax())


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
        return int(contradiction_share > entailment_share)
    elif contradiction_share >= contradiction_threshold:
        return 1
    elif entailment_share >= entailment_threshold:
        return 0
    return 2
