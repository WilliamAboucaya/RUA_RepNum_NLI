from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# For RUA datasets
consultation_data = pd.read_csv("consultation_data/rua-fonctionnement.csv", encoding='unicode_escape', engine='python', quoting=3, sep=';')
consultation_data["contributions_title"] = consultation_data["contributions_title"].fillna("")
consultation_data["contributions_bodyText"] = consultation_data["contributions_bodyText"].fillna("")
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]
sentences = proposals["contributions_title"].tolist()
paraphrases = util.paraphrase_mining(model, sentences)

# For RepNum dataset
# consultation_data = pd.read_csv("./consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
#                                 parse_dates=["Création", "Modification"],
#                                 index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
# consultation_data["Lié.à.."] = consultation_data["Lié.à.."].fillna("Unknown")
# consultation_data["Type.de.profil"] = consultation_data["Type.de.profil"].fillna("Unknown")
# proposals = consultation_data.loc[consultation_data["Type.de.contenu"] == "Proposition"]
# sentences = proposals["Titre"].tolist()
# paraphrases = util.paraphrase_mining(model, sentences)

with open('results/output_fr_fonctionnement.log', 'w', encoding='utf8') as f:
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        f.write("{} \t\t {} \t\t Score: {:.4f}\n".format(sentences[i], sentences[j], score))
