import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import importlib
import sys

from sknetwork.clustering import Louvain
from sknetwork.data import from_edge_list

sys.path.append('../')

from utils.functions import define_label


def generate_graph_from_dataframe(df: pd.DataFrame, label_col: str):
    edge_list = list(df.loc[float(df[label_col]) == 0][["premise_idx", "hypothesis_idx"]].itertuples(index=False))
    graph = from_edge_list(edge_list)

    return graph


def get_clusters_louvain(graph, modularity: str = "dugue"):
    cluster_labels = Louvain(modularity=modularity).fit_transform(graph.adjacency)

    return cluster_labels


if __name__ == "__main__":
    consultation_name = sys.argv[1]
    model_checkpoint = sys.argv[2]
    model_revision = sys.argv[3]
    strategy_to_apply = sys.argv[4]
    batch_size = int(sys.argv[5])

    model_name = model_checkpoint.split("/")[-1]

    if "contradictionshare" in strategy_to_apply:
        contradiction_threshold = float(sys.argv[6])
        entailment_threshold = float(sys.argv[7])

    strategy_to_apply_radix = "withpast_" + strategy_to_apply.split("_", 1)[1]
    apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply_radix}").apply_strategy

    proposals_couples = pd.read_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv", encoding="utf8", sep=';')

    result_column = f"{model_name}_{strategy_to_apply}_label"
    proposals_couples_labeled = pd.DataFrame(columns=['premise', 'premise_idx', 'hypothesis', 'hypothesis_idx', 'part', result_column])

    for part, df in proposals_couples.groupby("part"):
        df_labeled = apply_strategy(df, model_checkpoint, model_revision, batch_size)
        if "contradictionshare" in strategy_to_apply:
            df[result_column] = df_labeled.apply(
                lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"],
                                         contradiction_threshold, entailment_threshold), axis=1)
        else:
            df[result_column] = df_labeled["predicted_label"]

        proposals_couples_labeled = pd.concat([proposals_couples_labeled, df], ignore_index=True)
        proposals_couples_labeled.to_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}_flush.csv", sep=";", encoding="utf-8", index=False)

    proposals_couples_labeled.to_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv", sep=";", encoding="utf-8", index=False)

    # if not os.path.exists(f"../results/joblib_dumps/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
    #     os.makedirs(f"../results/joblib_dumps/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}", exist_ok=True)
    #
    # proposals_couples_by_part = proposals_couples_labeled.groupby("part")
    #
    # for part, df in proposals_couples_by_part:
    #     graph = generate_graph_from_dataframe(df, result_column)
    #     clusters_labels = get_clusters_louvain(graph)
    #
    #     unique, counts = np.unique(clusters_labels, return_counts=True)
    #
    #     clusters_sizes = dict(zip(unique, counts))
