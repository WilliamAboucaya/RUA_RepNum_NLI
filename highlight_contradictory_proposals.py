import nltk
import pandas as pd
import os

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

file_name = "rua-fonctionnement"

consultation_data = pd.read_csv(f"consultation_data/{file_name}.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
proposal_contents = proposals["contributions_bodyText"].tolist()

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]


def remove_past_sentences(proposal_content):
    sentences = sentences_tokenizer.tokenize(proposal_content)

    not_past_sentences = []

    for sentence in sentences:
        sentence_pos = nlp_token_class(sentence)
        if not any(token["entity"] in past_tense_tags for token in sentence_pos):
            not_past_sentences.append(sentence)

    return " ".join(not_past_sentences)


proposal_contents = list(filter(lambda proposal_content: proposal_content != "", proposal_contents))
proposal_contents = list(map(remove_past_sentences, proposal_contents))
proposal_contents = list(filter(lambda proposal_content: proposal_content != "", proposal_contents))

nli_model = AutoModelForSequenceClassification.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr")
nli_tokenizer = AutoTokenizer.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr", model_max_length=512)

if not os.path.exists(f"./results/{file_name}"):
    os.mkdir(f"./results/{file_name}")

with open(f"./results/{file_name}/contradictory_proposals_0.7.log", "w", encoding="utf8") as contradictory_file_dot7, \
     open(f"./results/{file_name}/contradictory_proposals_0.8.log", "w", encoding="utf8") as contradictory_file_dot8, \
     open(f"./results/{file_name}/contradictory_proposals_0.9.log", "w", encoding="utf8") as contradictory_file_dot9:
    for i in range(len(proposal_contents)):
        premise = proposal_contents[i]

        contradictory_proposals_dot7 = []
        contradictory_proposals_dot8 = []
        contradictory_proposals_dot9 = []

        for j in range(i + 1, len(proposal_contents)):
            hypothesis = proposal_contents[j]
            x = nli_tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=512, truncation=True)
            logits = nli_model(x)[0]
            probs = logits[:, ::].softmax(dim=1)
            prob_label_is_true = probs[:, 1]

            if prob_label_is_true > 0.9:
                contradictory_proposals_dot9.append(hypothesis)
            elif prob_label_is_true > 0.8:
                contradictory_proposals_dot8.append(hypothesis)
            elif prob_label_is_true > 0.7:
                contradictory_proposals_dot7.append(hypothesis)

        for (contradictory_proposals, contradictory_file) in zip([contradictory_proposals_dot7, contradictory_proposals_dot8, contradictory_proposals_dot9],
                                                                 [contradictory_file_dot7, contradictory_file_dot8, contradictory_file_dot9]):
            if contradictory_proposals:
                contradictory_file.write(f"{hypothesis}\n\n")
                for proposal in contradictory_proposals:
                    contradictory_file.write(f"{proposal}\n")

                contradictory_file.write("==================================\n\n")
