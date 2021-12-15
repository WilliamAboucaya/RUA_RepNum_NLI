from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk

from collections import Counter

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# For RepNum dataset
# consultation_data = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
#                                 parse_dates=["Création", "Modification"],
#                                 index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
# consultation_data["Lié.à.."] = consultation_data["Lié.à.."].fillna("Unknown")
# consultation_data["Type.de.profil"] = consultation_data["Type.de.profil"].fillna("Unknown")
#
# proposals = consultation_data.loc[consultation_data["Type.de.contenu"] == "Proposition"]
# proposal_titles = proposals["Titre"].tolist()
# proposal_contents = proposals["Contenu"].tolist()

# For RUA datasets
consultation_data = pd.read_csv("consultation_data/rua-publics.csv", engine='python', quoting=3, sep=';')
consultation_data["contributions_title"] = consultation_data["contributions_title"].fillna("")
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
proposal_titles = proposals["contributions_title"].tolist()
proposal_contents = proposals["contributions_bodyText"].tolist()

paraphrases = util.paraphrase_mining(model, proposal_contents)

paraphrase_scores = [{} for i in range(len(proposal_contents))]

for paraphrase in paraphrases:
    score, i, j = paraphrase

    paraphrase_scores[i][j] = score
    paraphrase_scores[j][i] = score

with open('results/proposals_content_comparison_publics_clean.txt', 'w', encoding='utf8') as f:
    for idx, sentence_paraphrase_scores in enumerate(paraphrase_scores):
        k = Counter(sentence_paraphrase_scores)
        highest_scores = k.most_common(5)

        f.write(f"Initial proposal:\n{proposal_contents[idx]}\nClosest neighbors:\n")

        for score in highest_scores:
            f.write(f"{proposal_contents[score[0]]}: {score[1]}\n")
        f.write("\n")
