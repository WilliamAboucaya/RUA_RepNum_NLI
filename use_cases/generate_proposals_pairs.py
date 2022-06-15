import pandas as pd

# repnum_consultation = pd.read_csv("../consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
#                                   parse_dates=["Création", "Modification"],
#                                   index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
# repnum_consultation["Lié.à.."] = repnum_consultation["Lié.à.."].fillna("Unknown")
# repnum_consultation["Type.de.profil"] = repnum_consultation["Type.de.profil"].fillna("Unknown")
# repnum_proposals = repnum_consultation.loc[repnum_consultation["Type.de.contenu"] == "Proposition"]
# repnum_proposals["full_proposal"] = repnum_proposals.apply(lambda row: row["Titre"] + ". " + row["Contenu"], axis=1)
#
# repnum_proposals_pairs = pd.DataFrame(columns=['premise', 'premise_idx', 'hypothesis', 'hypothesis_idx'])
#
# for idx, premise in repnum_proposals["full_proposal"].iteritems():
#     for idx2, hypothesis in repnum_proposals.iloc[idx + 1:]["full_proposal"].iteritems():
#         formatted_row = pd.DataFrame({'premise': [premise], 'premise_idx': [idx], 'hypothesis': [hypothesis], 'hypothesis_idx': [idx2]})
#         repnum_proposals_pairs = pd.concat([repnum_proposals_pairs, formatted_row], ignore_index=True)
#
# repnum_proposals_pairs.to_csv("../consultation_data/proposals_pairs_repnum.csv", sep=";", encoding="utf-8", index=False)
#
# del(repnum_consultation, repnum_proposals, repnum_proposals_pairs)

rua_consultation_1 = pd.read_csv("../consultation_data/rua-fonctionnement.csv", encoding="utf8",  engine='python', sep=';')
rua_consultation_1["contributions_title"] = rua_consultation_1["contributions_title"].fillna("")
rua_consultation_1["contributions_bodyText"] = rua_consultation_1["contributions_bodyText"].fillna("")
rua_consultation_2 = pd.read_csv("../consultation_data/rua-principes.csv", encoding="utf8",  engine='python', sep=';')
rua_consultation_2["contributions_title"] = rua_consultation_2["contributions_title"].fillna("")
rua_consultation_2["contributions_bodyText"] = rua_consultation_2["contributions_bodyText"].fillna("")
rua_consultation_3 = pd.read_csv("../consultation_data/rua-publics.csv", encoding="utf8",  engine='python', sep=';')
rua_consultation_3["contributions_title"] = rua_consultation_3["contributions_title"].fillna("")
rua_consultation_3["contributions_bodyText"] = rua_consultation_3["contributions_bodyText"].fillna("")
rua_consultation = rua_consultation_1.append([rua_consultation_2, rua_consultation_3])
rua_proposals = rua_consultation.loc[rua_consultation["type"] == "opinion"].reset_index(drop=True)
rua_proposals["full_proposal"] = rua_proposals.apply(lambda row: row["contributions_title"] + ". " + row["contributions_bodyText"], axis=1)

rua_proposals_pairs = pd.DataFrame(columns=['premise', 'premise_idx', 'hypothesis', 'hypothesis_idx'])

for idx, premise in rua_proposals["full_proposal"].iteritems():
    for idx2, hypothesis in rua_proposals.iloc[idx + 1:]["full_proposal"].iteritems():
        formatted_row = pd.DataFrame({'premise': [premise], 'premise_idx': [idx], 'hypothesis': [hypothesis], 'hypothesis_idx': [idx2]})
        rua_proposals_pairs = pd.concat([rua_proposals_pairs, formatted_row], ignore_index=True)

rua_proposals_pairs.to_csv("../consultation_data/proposals_pairs_rua.csv", sep=";", encoding="utf-8", index=False)
