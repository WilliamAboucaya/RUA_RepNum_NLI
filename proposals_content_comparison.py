from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk

from collections import Counter

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# For RepNum dataset
consultation_data = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                                parse_dates=["Création", "Modification"],
                                index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
consultation_data["Lié.à.."] = consultation_data["Lié.à.."].fillna("Unknown")
consultation_data["Type.de.profil"] = consultation_data["Type.de.profil"].fillna("Unknown")

proposals = consultation_data.loc[consultation_data["Type.de.contenu"] == "Proposition"]
proposal_titles = proposals["Titre"].tolist()
proposal_contents = proposals["Contenu"].tolist()

# For RUA datasets
# consultation_data_1 = pd.read_csv("consultation_data/rua-fonctionnement.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
# consultation_data_1["contributions_title"] = consultation_data_1["contributions_title"].fillna("")
# consultation_data_1["contributions_bodyText"] = consultation_data_1["contributions_bodyText"].fillna("")
#
# consultation_data_2 = pd.read_csv("consultation_data/rua-principes.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
# consultation_data_2["contributions_title"] = consultation_data_2["contributions_title"].fillna("")
# consultation_data_2["contributions_bodyText"] = consultation_data_2["contributions_bodyText"].fillna("")
#
# consultation_data_3 = pd.read_csv("consultation_data/rua-publics.csv", encoding="utf8",  engine='python', quoting=3, sep=';')
# consultation_data_3["contributions_title"] = consultation_data_3["contributions_title"].fillna("")
# consultation_data_3["contributions_bodyText"] = consultation_data_3["contributions_bodyText"].fillna("")
#
# proposals_1 = consultation_data_1.loc[consultation_data_1["type"] == "opinion"]
# proposals_2 = consultation_data_2.loc[consultation_data_2["type"] == "opinion"]
# proposals_3 = consultation_data_3.loc[consultation_data_3["type"] == "opinion"]
# proposal_titles = proposals_1["contributions_title"].append([proposals_2["contributions_title"], proposals_3["contributions_title"]]).tolist()
# proposal_contents = proposals_1["contributions_bodyText"].append([proposals_2["contributions_bodyText"], proposals_3["contributions_bodyText"]]).tolist()
#
# p2_start = len(proposals_1.index)
# p3_start = p2_start + len(proposals_2.index)

paraphrases = util.paraphrase_mining(model, proposal_contents)

paraphrase_scores = [{} for i in range(len(proposal_contents))]

for paraphrase in paraphrases:
    score, i, j = paraphrase

    paraphrase_scores[i][j] = score
    paraphrase_scores[j][i] = score

with open('results/proposals_content_comparison_repnum.txt', 'w', encoding='utf8') as f:
    for idx, sentence_paraphrase_scores in enumerate(paraphrase_scores):
        k = Counter(sentence_paraphrase_scores)
        highest_scores = k.most_common(5)

        # RepNum
        f.write(f"Initial proposal:\n{proposal_contents[idx]}\nClosest neighbors:\n")
        # RUA
        # f.write(f"Initial proposal:\n*{'fonctionnement' if idx < p2_start else 'principes' if idx < p3_start else 'publics'}*:{proposal_contents[idx]}\nClosest neighbors:\n")

        for score in highest_scores:
            # RepNum
            f.write(f"{proposal_contents[score[0]]}: {score[1]}\n")
            # RUA
            # f.write(f"*{'fonctionnement' if score[0] < p2_start else 'principes' if score[0 < p3_start] else 'publics'}*: {proposal_contents[score[0]]}: {score[1]}\n")
        f.write("\n")
