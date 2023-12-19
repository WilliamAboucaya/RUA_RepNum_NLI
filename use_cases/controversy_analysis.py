import argparse

import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm


def get_proposal_stats(idx, proposals_pairs_df, consultation_data_df, result_column_label, consultation):
    relevant_pairs = proposals_pairs_df.loc[(proposals_pairs_df['premise_idx'] == idx) | (proposals_pairs_df['hypothesis_idx'] == idx)]
    proposal_text = relevant_pairs.iloc[0]["premise"] if relevant_pairs.iloc[0]["premise_idx"] == idx else relevant_pairs.iloc[0]["hypothesis"]

    nb_entailed_proposals = len(relevant_pairs.loc[proposals_pairs_df[result_column_label] == 0].index)
    nb_contradictory_proposals = len(relevant_pairs.loc[proposals_pairs_df[result_column_label] == 1].index)

    if "repnum" in consultation:
        proposals_data_df = consultation_data_df.loc[consultation_data_df["Type.de.contenu"] == "Proposition"]
        proposal_id = proposals_data_df.loc[(proposals_data_df["Titre"] + ". " + proposals_data_df["Contenu"]) == proposal_text].iloc[0]["Identifiant"]
        proposal = proposals_data_df.loc[proposals_data_df["Identifiant"] == proposal_id].iloc[0]

        proposal_creation_datetime = proposal["Création"]
        nb_votes_for = proposal["Votes.pour"]
        nb_votes_mixed = proposal["Votes.mitigés"]
        nb_votes_against = proposal["Votes.contre"]
        nb_args_for = proposal["Arguments.pour"]
        nb_args_against = proposal["Arguments.contre"]

    elif "rua" in consultation:
        proposals_data_df = consultation_data_df.loc[consultation_data_df["type"] == "opinion"]
        proposal_id = proposals_data_df.loc[(proposals_data_df["contributions_title"] + ". " + proposals_data_df["contributions_bodyText"]) == proposal_text].iloc[0]["contributions_id"]
        proposal = proposals_data_df.loc[proposals_data_df["contributions_id"] == proposal_id].iloc[0]

        proposal_creation_datetime = proposal["contributions_createdAt"]
        nb_votes_for = len(consultation_data_df.loc[(consultation_data_df["contributions_votes_related_id"] == proposal_id) & (consultation_data_df["contributions_votes_value"] == "YES")].index)
        nb_votes_mixed = len(consultation_data_df.loc[(consultation_data_df["contributions_votes_related_id"] == proposal_id) & (consultation_data_df["contributions_votes_value"] == "MITIGE")].index)
        nb_votes_against = len(consultation_data_df.loc[(consultation_data_df["contributions_votes_related_id"] == proposal_id) & (consultation_data_df["contributions_votes_value"] == "NO")].index)
        nb_args_for = len(consultation_data_df.loc[(consultation_data_df["contributions_arguments_related_id"] == proposal_id) & (consultation_data_df["contributions_arguments_type"] == "FOR")].index)
        nb_args_against = len(consultation_data_df.loc[(consultation_data_df["contributions_arguments_related_id"] == proposal_id) & (consultation_data_df["contributions_arguments_type"] == "AGAINST")].index)
    else:
        proposal_id = proposal_creation_datetime = nb_entailed_proposals = nb_contradictory_proposals = nb_votes_for = nb_votes_mixed = nb_votes_against = nb_args_for = nb_args_against = 0

    return proposal_id, proposal_text, proposal_creation_datetime, nb_entailed_proposals, nb_contradictory_proposals, nb_votes_for, nb_votes_mixed, nb_votes_against, nb_args_for, nb_args_against


def compute_proposal_metrics(nb_entailed_proposals, nb_contradictory_proposals, nb_votes_for, nb_votes_mixed, nb_votes_against, nb_args_for, nb_args_against):
    if (nb_entailed_proposals + nb_contradictory_proposals) > 20:
        controversiality_nli = (min(nb_entailed_proposals, nb_contradictory_proposals) + 1) / (max(nb_entailed_proposals, nb_contradictory_proposals) + 1)
    else:
        controversiality_nli = -1

    if (nb_votes_for + nb_votes_mixed + nb_votes_against) > 15:
        controversiality_votes = (min(nb_votes_for, nb_votes_against) + nb_votes_mixed + 1) / (max(nb_votes_for, nb_votes_against) + nb_votes_mixed + 1)
    else:
        controversiality_votes = -1

    if (nb_args_for + nb_args_against) > 8:
        controversiality_args = (min(nb_args_for, nb_args_against) + 1) / (max(nb_args_for, nb_args_against) + 1)
    else:
        controversiality_args = -1

    return controversiality_nli, controversiality_votes, controversiality_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("consultation_name", type=str)
    parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("strategy_to_apply", type=str)
    args = parser.parse_args()

    consultation_name: str = args.consultation_name
    model_checkpoint: str = args.model_checkpoint
    strategy_to_apply: str = args.strategy_to_apply

    if "repnum" in consultation_name:
        consultation_data = pd.read_csv("../consultation_data/projet-de-loi-numerique-consultation-anonyme.csv",
                                        parse_dates=["Création", "Modification"],
                                        index_col=0, dtype={"Identifiant": str, "Titre": str, "Lié.à..": str, "Contenu": str, "Lien": str})
        consultation_data["Lié.à.."] = consultation_data["Lié.à.."].fillna("Unknown")
        consultation_data["Type.de.profil"] = consultation_data["Type.de.profil"].fillna("Unknown")
    elif "rua" in consultation_name:
        consultation_1 = pd.read_csv("../consultation_data/rua-fonctionnement.csv", encoding="utf8", engine='python', sep=';')
        consultation_1["contributions_title"] = consultation_1["contributions_title"].fillna("")
        consultation_1["contributions_bodyText"] = consultation_1["contributions_bodyText"].fillna("")
        consultation_2 = pd.read_csv("../consultation_data/rua-principes.csv", encoding="utf8", engine='python', sep=';')
        consultation_2["contributions_title"] = consultation_2["contributions_title"].fillna("")
        consultation_2["contributions_bodyText"] = consultation_2["contributions_bodyText"].fillna("")
        consultation_3 = pd.read_csv("../consultation_data/rua-publics.csv", encoding="utf8", engine='python', sep=';')
        consultation_3["contributions_title"] = consultation_3["contributions_title"].fillna("")
        consultation_3["contributions_bodyText"] = consultation_3["contributions_bodyText"].fillna("")
        consultation_data = pd.concat([consultation_1, consultation_2, consultation_3], ignore_index=True)
        consultation_data["contributions_createdAt"] = pd.to_datetime(consultation_data["contributions_createdAt"])
    else:
        exit()

    model_name = model_checkpoint.split("/")[-1]

    proposals_couples = pd.read_csv(
        f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv",
        encoding="utf8", sep=';')

    result_column = f"{model_name}_{strategy_to_apply}_label"

    proposals_idxs = pd.unique(proposals_couples[['premise_idx', 'hypothesis_idx']].values.ravel('K'))

    controversy_data_dict = {"proposal_id": [], "proposal_text": [], "creation_datetime": [],
                             "nb_entailed_proposals": [], "nb_contradictory_proposals": [], "nb_votes_for": [],
                             "nb_votes_mixed": [], "nb_votes_against": [], "nb_args_for": [], "nb_args_against": []}

    print("Retrieving data for proposals...")

    for proposal_idx in tqdm(proposals_idxs):
        controversy_data = get_proposal_stats(proposal_idx, proposals_couples, consultation_data, result_column, consultation_name)

        controversy_data_dict["proposal_id"].append(controversy_data[0])
        controversy_data_dict["proposal_text"].append(controversy_data[1])
        controversy_data_dict["creation_datetime"].append(controversy_data[2])
        controversy_data_dict["nb_entailed_proposals"].append(controversy_data[3])
        controversy_data_dict["nb_contradictory_proposals"].append(controversy_data[4])
        controversy_data_dict["nb_votes_for"].append(controversy_data[5])
        controversy_data_dict["nb_votes_mixed"].append(controversy_data[6])
        controversy_data_dict["nb_votes_against"].append(controversy_data[7])
        controversy_data_dict["nb_args_for"].append(controversy_data[8])
        controversy_data_dict["nb_args_against"].append(controversy_data[9])

    controversy_df = pd.DataFrame(data=controversy_data_dict)

    print("Computing controversy metrics...")

    controversy_df[["controversiality_nli", "controversiality_votes", "controversiality_args"]] = controversy_df.apply(
        lambda row: compute_proposal_metrics(row["nb_entailed_proposals"], row["nb_contradictory_proposals"],
                                             row["nb_votes_for"], row["nb_votes_mixed"], row["nb_votes_against"],
                                             row["nb_args_for"], row["nb_args_against"]), axis=1, result_type="expand"
    )

    controversy_df.to_csv(f"../results/controversy_measures_{consultation_name}_{strategy_to_apply}.csv", sep=",", encoding="utf-8", index=False)

    controversy_nli_votes = controversy_df.loc[(controversy_df["controversiality_nli"] != -1) & (controversy_df["controversiality_votes"] != -1)]
    pearson_nli_votes = pearsonr(controversy_nli_votes["controversiality_nli"].tolist(), controversy_nli_votes["controversiality_votes"].tolist())[0]

    controversy_nli_args = controversy_df.loc[(controversy_df["controversiality_nli"] != -1) & (controversy_df["controversiality_args"] != -1)]
    pearson_nli_args = pearsonr(controversy_nli_args["controversiality_nli"].tolist(), controversy_nli_args["controversiality_args"].tolist())[0]

    controversy_votes_args = controversy_df.loc[(controversy_df["controversiality_votes"] != -1) & (controversy_df["controversiality_args"] != -1)]
    pearson_votes_args = pearsonr(controversy_votes_args["controversiality_votes"].tolist(), controversy_votes_args["controversiality_args"].tolist())[0]

    print(f"Pearson correlation between controversiality for NLI and votes: {pearson_nli_votes}")
    print(f"Pearson correlation between controversiality for NLI and args: {pearson_nli_args}")
    print(f"Pearson correlation between controversiality for votes and args: {pearson_votes_args}")

    controversy_q3_nli = controversy_df["controversiality_nli"].loc[controversy_df["controversiality_nli"] > -1].quantile(0.75, interpolation="lower")
    controversy_q3_votes = controversy_df["controversiality_votes"].loc[controversy_df["controversiality_votes"] > -1].quantile(0.75, interpolation="lower")
    controversy_q3_args = controversy_df["controversiality_args"].loc[controversy_df["controversiality_args"] > -1].quantile(0.75, interpolation="lower")

    controversy_df_q3_nli = controversy_df.loc[(controversy_df["controversiality_nli"] >= controversy_q3_nli) &
                                               (pd.notna(controversy_df["creation_datetime"]))]
    controversy_df_q3_votes = controversy_df.loc[(controversy_df["controversiality_votes"] >= controversy_q3_votes) &
                                                 (pd.notna(controversy_df["creation_datetime"]))]
    controversy_df_q3_args = controversy_df.loc[(controversy_df["controversiality_args"] >= controversy_q3_args) &
                                                (pd.notna(controversy_df["creation_datetime"]))]

    med_date_nli = str(controversy_df_q3_nli["creation_datetime"].quantile(0.5, interpolation="lower"))
    med_date_votes = str(controversy_df_q3_votes["creation_datetime"].quantile(0.5, interpolation="lower"))
    med_date_args = str(controversy_df_q3_args["creation_datetime"].quantile(0.5, interpolation="lower"))

    print(f"Median date of creation for the proposals among the 25 % highest controversiality for NLI: {med_date_nli}")
    print(f"Median date of creation for the proposals among the 25 % highest controversiality for votes: {med_date_votes}")
    print(f"Median date of creation for the proposals among the 25 % highest controversiality for args: {med_date_args}")
