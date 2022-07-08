import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

import importlib
import sys

from utils.functions import define_label

sys.path.append('../')


def generate_graph_from_dataframe(df: pd.DataFrame, strategy: str, model_checkpoint: str, model_revision: str = "main") -> nx.Graph:
    graph = nx.Graph()

    apply_strategy = importlib.import_module(f"contradiction_checking.{strategy}").apply_strategy

    df_labeled = apply_strategy(df, model_checkpoint, model_revision)

    graph.add_nodes_from(pd.concat([df_labeled["premise_idx"], df_labeled["hypothesis_idx"]]).unique())

    df_labeled.apply(lambda row: graph.add_edge(row["premise_idx"], row["hypothesis_idx"]) if row["predicted_label"] == 0 else None)

    return graph


if __name__ == "__main__":
    consultation_name = sys.argv[1]
    strategy_to_apply = sys.argv[2]
    model_checkpoint = sys.argv[3]
    model_revision = sys.argv[4]
    batch_size = int(sys.argv[5])

    model_name = model_checkpoint.split("/")[-1]

    if "contradictionshare" in strategy_to_apply:
        contradiction_threshold = float(sys.argv[6])
        entailment_threshold = float(sys.argv[7])

    strategy_to_apply_radix = "withpast_" + strategy_to_apply.split("_", 1)[1]
    apply_strategy = importlib.import_module(f"contradiction_checking.{strategy_to_apply_radix}").apply_strategy

    proposals_couples = pd.read_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv")

    proposals_couples_labeled = apply_strategy(proposals_couples, model_checkpoint, model_revision, batch_size)

    result_column = f"{model_name}_{strategy_to_apply}_label"
    if "contradictionshare" in strategy_to_apply:
        proposals_couples[result_column] = proposals_couples_labeled.apply(
            lambda row: define_label(row["share_contradictory_pairs"], row["share_entailed_pairs"], contradiction_threshold, entailment_threshold), axis=1)
    else:
        proposals_couples[result_column] = proposals_couples_labeled["predicted_label"]

    proposals_couples.to_csv(f"../consultation_data/proposals_pairs_{consultation_name}{'_nopast' if 'removepast' in strategy_to_apply else ''}.csv")

    # if not os.path.exists(f"../results/joblib_dumps/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
    #     os.makedirs(f"../results/joblib_dumps/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}", exist_ok=True)

    # proposals_pairs_by_part = proposals_pairs.groupby("part")
    #
    # for part, df in proposals_pairs_by_part:
    #     graph = generate_graph_from_dataframe(repnum_consultation, strategy_to_apply_radix, model_checkpoint, model_revision)
    #     joblib.dump(graph, f"../results/joblib_dumps/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/repnum_graph_{strategy_to_apply}.joblib")
    #
    # nx.draw(repnum_graph)
    # plt.show()
