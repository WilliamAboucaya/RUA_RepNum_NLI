import treetaggerwrapper as ttwp
import nltk
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

consultation_data = pd.read_csv("consultation_data/rua-fonctionnement.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
proposal_contents = proposals["contributions_bodyText"].tolist()

tt_fr = ttwp.TreeTagger(TAGLANG="fr")
sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
camembert_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=camembert_tokenizer)

past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]

with open("results/comparison-tt-tf-nopast-fonctionnement-contents.log", "w", encoding="utf8") as nopast:

    proposals_modified_tt = 0
    proposals_modified_tf = 0

    for proposal_content in proposal_contents:
        if proposal_content != "":
            sentences = sentences_tokenizer.tokenize(proposal_content)

            not_past_sentences = {"treetagger": [], "transformers": []}

            for sentence in sentences:
                sentence_pos_tt = ttwp.make_tags(tt_fr.tag_text(sentence), exclude_nottags=True)
                if not any(token.pos in past_tense_tags for token in sentence_pos_tt):
                    not_past_sentences["treetagger"].append(sentence)

                sentence_pos_transformers = nlp_token_class(sentence)
                if not any(token["entity"] in past_tense_tags for token in sentence_pos_transformers):
                    not_past_sentences["transformers"].append(sentence)

            nopast.write(f'{proposal_content}\n')

            if proposal_content != " ".join(not_past_sentences["treetagger"]) and \
               " ".join(not_past_sentences["treetagger"]) == " ".join(not_past_sentences["transformers"]):
                nopast.write(f'TT/TF: {" ".join(not_past_sentences["treetagger"])}\n')
                proposals_modified_tt += 1
                proposals_modified_tf += 1

            else:
                if proposal_content != " ".join(not_past_sentences["treetagger"]):
                    nopast.write(f'TT: {" ".join(not_past_sentences["treetagger"])}\n')
                    proposals_modified_tt += 1

                if proposal_content != " ".join(not_past_sentences["transformers"]):
                    nopast.write(f'TF: {" ".join(not_past_sentences["transformers"])}\n')
                    proposals_modified_tf += 1

            nopast.write("=============================\n\n")

    nopast.write("Nb of proposals with past tense sentences:\n")
    nopast.write(f"According to TreeTagger: {proposals_modified_tt}\n")
    nopast.write(f"According to Transformers: {proposals_modified_tf}\n")
