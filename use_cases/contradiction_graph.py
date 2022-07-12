import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import importlib
import sys

from sknetwork.clustering import Louvain, modularity
from sknetwork.data import from_edge_list

sys.path.append('../')

from utils.functions import define_label


def generate_graph_from_dataframe(df: pd.DataFrame, label_col: str):
    edge_list = list(df.loc[df[label_col].astype(float) == 0][["premise", "hypothesis"]].itertuples(index=False))
    graph = from_edge_list(edge_list)

    return graph


def get_clusters_louvain(graph, resolution: float = 1):
    cluster_labels = Louvain(modularity="newman", resolution=resolution).fit_transform(graph.adjacency)
    cluster_modularity = modularity(graph.adjacency, cluster_labels, resolution=resolution)

    return cluster_labels, cluster_modularity


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

    apply_nli = "--apply-nli" in sys.argv

    strategy_to_apply_radix = "withpast_" + strategy_to_apply.split("_", 1)[1]
    apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply_radix}").apply_strategy

    proposals_couples = pd.read_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv", encoding="utf8", sep=';')

    result_column = f"{model_name}_{strategy_to_apply}_label"
    if apply_nli:
        proposals_couples_labeled = pd.DataFrame(columns=[*proposals_couples.columns[1:], result_column])

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
    else:
        proposals_couples_labeled = proposals_couples

    if not os.path.exists(f"../results/use_case/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
        os.makedirs(f"../results/use_case/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}", exist_ok=True)

    proposals_couples_by_part = proposals_couples_labeled.groupby("part")

    with open(f"../results/use_case/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/{strategy_to_apply}.log", "w", encoding="utf8") as file:
        for part, df in proposals_couples_by_part:
            graph = generate_graph_from_dataframe(df, result_column)
            clusters_labels, clusters_modularity = get_clusters_louvain(graph, resolution=1.5)

            unique, counts = np.unique(clusters_labels, return_counts=True)
            clusters_sizes = dict(zip(unique, counts))

            file.write(f"For part {part} with modularity {clusters_modularity}, clusters are:\n")
            for key, value in clusters_sizes.items():
                file.write(f"Cluster {key}: {value} proposals\n")
