import nltk
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline

consultation_data = pd.read_csv("consultation_data/rua-fonctionnement.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
proposal_contents = proposals["contributions_bodyText"].tolist()

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
pos_tokenizer = AutoTokenizer.from_pretrained("waboucay/french-camembert-postag-model-finetuned-perceo")
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

past_tense_tags = ["VER:impf", "VER:simp", "VER:subi"]

nli_model = AutoModelForSequenceClassification.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr")
nli_tokenizer = AutoTokenizer.from_pretrained("waboucay/camembert-base-finetuned-xnli_fr")
