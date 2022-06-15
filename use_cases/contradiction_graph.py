import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

import importlib


def generate_graph_from_dataframe(df: pd.DataFrame, strategy: str, model_checkpoint: str, model_revision: str = "main") -> nx.Graph:
    graph = nx.Graph()

    apply_strategy = importlib.import_module(f"contradiction_checking.{strategy}").apply_strategy

    df_labeled = apply_strategy(df, model_checkpoint, model_revision)

    graph.add_nodes_from(df_labeled["premise_idx"].unique())

    df_labeled.apply(lambda row: graph.add_edge(row["premise_idx"], row["hypothesis_idx"]) if row["predicted_label"] == 0 else None)

    return graph


strategy_to_apply = "withpast_proposalwise"
input_model_checkpoint = "waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl"
input_model_revision = "main"
input_model_name = input_model_checkpoint.split("/")[-1]

if not os.path.exists(f"../results/joblib_dumps/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}"):
    os.mkdir(f"../results/joblib_dumps/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}")

repnum_consultation = pd.read_csv("../consultation_data/proposals_pairs_repnum.csv")
repnum_graph = generate_graph_from_dataframe(repnum_consultation, strategy_to_apply, input_model_checkpoint, input_model_revision)

joblib.dump(repnum_graph, f"../results/joblib_dumps/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/repnum_graph_{strategy_to_apply}.joblib")

nx.draw(repnum_graph)
plt.show()

del(repnum_consultation, repnum_graph)

rua_consultation = pd.read_csv("../consultation_data/proposals_pairs_rua.csv")
rua_graph = generate_graph_from_dataframe(rua_consultation, strategy_to_apply, input_model_checkpoint, input_model_revision)

joblib.dump(rua_graph, f"../results/joblib_dumps/{input_model_name}{('_' + input_model_revision) if input_model_revision != 'main' else ''}/rua_graph_{strategy_to_apply}.joblib")

nx.draw(rua_graph)
plt.show()
