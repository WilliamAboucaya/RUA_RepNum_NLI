import sys

import pandas as pd

sys.path.append('../')
from utils.functions import generate_proposals_pairs_repnum, generate_proposals_pairs_rua

if __name__ == "__main__":
    repnum_consultation = pd.read_csv("../consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                                      parse_dates=["Création", "Modification"],
                                      index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
    repnum_consultation["Lié.à.."] = repnum_consultation["Lié.à.."].fillna("Unknown")
    repnum_consultation["Type.de.profil"] = repnum_consultation["Type.de.profil"].fillna("Unknown")
    repnum_proposals = repnum_consultation.loc[repnum_consultation["Type.de.contenu"] == "Proposition"].reset_index(drop=True)

    repnum_proposals_pairs = generate_proposals_pairs_repnum(repnum_proposals)
    repnum_proposals_pairs.to_csv("../consultation_data/proposals_pairs_repnum.csv", sep=";", encoding="utf-8", index=False)

    repnum_proposals_nopast_pairs = generate_proposals_pairs_repnum(repnum_proposals, no_past=True)
    repnum_proposals_nopast_pairs.to_csv("../consultation_data/proposals_pairs_repnum_nopast.csv", sep=";", encoding="utf-8", index=False)

    del(repnum_consultation, repnum_proposals, repnum_proposals_pairs, repnum_proposals_nopast_pairs)

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

    rua_proposals_pairs = generate_proposals_pairs_rua(rua_proposals)
    rua_proposals_pairs.to_csv("../consultation_data/proposals_pairs_rua.csv", sep=";", encoding="utf-8", index=False)

    rua_proposals_nopast_pairs = generate_proposals_pairs_rua(rua_proposals, no_past=True)
    rua_proposals_nopast_pairs.to_csv("../consultation_data/proposals_pairs_rua_nopast.csv", sep=";", encoding="utf-8", index=False)
