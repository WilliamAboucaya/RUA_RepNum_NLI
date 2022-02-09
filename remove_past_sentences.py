import treetaggerwrapper as ttwp
import nltk
import pandas as pd


consultation_data = pd.read_csv("consultation_data/rua-publics.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
proposal_contents = proposals["contributions_bodyText"].tolist()

tt_fr = ttwp.TreeTagger(TAGLANG="fr")
french_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")

past_tense_tags = ["VER:impf", "VER:pper", "VER:simp", "VER:subi"]

proposals_clean = []

for proposal_content in proposal_contents:
    if proposal_content != "":
        sentences = french_tokenizer.tokenize(proposal_content)

        not_past_sentences = []
        for sentence in sentences:
            sentence_pos = ttwp.make_tags(tt_fr.tag_text(sentence))
            if not any(pos.pos in past_tense_tags for pos in sentence_pos):
                not_past_sentences.append(sentence)

        proposals_clean.append(" ".join(not_past_sentences))

with open("results/nopast-publics-contents.txt", "w", encoding="utf8") as nopast:
    for proposal in proposals_clean:
        if proposal != "":
            nopast.write(proposal + "\n")
