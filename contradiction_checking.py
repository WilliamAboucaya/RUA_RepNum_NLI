import nltk
import pandas as pd

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

labeled_proposals_couples = pd.read_csv("consultation_data/nli_labeled_proposals.csv", encoding="utf8",  engine='python', quoting=3, sep=';')

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]

nli_model = AutoModelForSequenceClassification.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr")
nli_tokenizer = AutoTokenizer.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr", model_max_length=512)


def remove_past_sentences(proposal_content):
    sentences = sentences_tokenizer.tokenize(proposal_content)
    not_past_sentences = []

    for sentence in sentences:
        sentence_pos = nlp_token_class(sentence)
        if not any(token["entity"] in past_tense_tags for token in sentence_pos):
            not_past_sentences.append(sentence)

    return " ".join(not_past_sentences)


def predict_nli(premise, hypothesis):
    x = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=512, truncation=True)
    logits = nli_model(x)[0]
    probs = logits[:, ::].softmax(dim=1)
    return float(probs[:, 1])


labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(remove_past_sentences)
labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(remove_past_sentences)

with open("./results/contradiction_checking.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        predicted_label = predict_nli(row["premise"], row["hypothesis"])

        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Prediction: {predicted_label};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")
