import os
import re

consultation_name = "repnum"
without_alternate_thresholds = True
min_recall = 0.2

f1_results = {}
precision_results_0 = {}
precision_results_1 = {}
precision_results_2 = {}
recall_results_0 = {}
recall_results_1 = {}
recall_results_2 = {}

for model_name in os.listdir(f"../results/contradiction_checking/{consultation_name}"):
    if os.path.isdir(f"../results/contradiction_checking/{consultation_name}/{model_name}") and "large" in model_name and "3_classes" in model_name:
        for file_name in os.listdir(f"../results/contradiction_checking/{consultation_name}/{model_name}"):
            if "metrics" in file_name:
                with open(f"../results/contradiction_checking/{consultation_name}/{model_name}/{file_name}", "r") as file:
                    if "contradictionshare" in file_name:
                        offset = 1
                    else:
                        offset = 0

                    for i, line in enumerate(file):
                        if i % (4 + offset) == 0:
                            if offset == 1:
                                threshold_contradictory, threshold_entailed = [float(t) for t in re.findall("\d\.\d+", line)]
                                key = f"{model_name} {file_name} {threshold_contradictory} {threshold_entailed}"
                            else:
                                key = f"{model_name} {file_name}"
                        if i % (4 + offset) == (0 + offset):
                            precisions_list = re.findall("\d\.\d+", line)
                            precision_results_0[key] = float(precisions_list[0])
                            precision_results_1[key] = float(precisions_list[1])
                            precision_results_2[key] = float(precisions_list[2])
                        if i % (4 + offset) == (1 + offset):
                            recall_list = re.findall("\d\.\d+", line)
                            if float(recall_list[0]) > min_recall:
                                recall_results_0[key] = float(recall_list[0])
                            else:
                                del precision_results_0[key]
                            if float(recall_list[1]) > min_recall:
                                recall_results_1[key] = float(recall_list[1])
                            else:
                                del precision_results_1[key]
                            if float(recall_list[2]) > min_recall:
                                recall_results_2[key] = float(recall_list[2])
                            else:
                                del precision_results_2[key]
                        if i % (4 + offset) == (3 + offset):
                            f1_results[key] = float(re.findall("\d+\.\d+", line)[0])
                            if without_alternate_thresholds:
                                break

f1_results_sorted = sorted(f1_results.items(), key=lambda x: x[1], reverse=True)
precision_results_sorted_0 = sorted(precision_results_0.items(), key=lambda x: x[1], reverse=True)
precision_results_sorted_1 = sorted(precision_results_1.items(), key=lambda x: x[1], reverse=True)
precision_results_sorted_2 = sorted(precision_results_2.items(), key=lambda x: x[1], reverse=True)
recall_results_sorted_0 = sorted(recall_results_0.items(), key=lambda x: precision_results_0[x[0]], reverse=True)
recall_results_sorted_1 = sorted(recall_results_1.items(), key=lambda x: precision_results_1[x[0]], reverse=True)
recall_results_sorted_2 = sorted(recall_results_2.items(), key=lambda x: precision_results_2[x[0]], reverse=True)

with open(f"../results/contradiction_checking/{consultation_name}/assessment_results{'_extended' if not without_alternate_thresholds else ''}.log", "w") as file:
    file.write(f"Best F1 macro:\n{f1_results_sorted[0][0]}: {f1_results_sorted[0][1]}\n{f1_results_sorted[1][0]}: {f1_results_sorted[1][1]}\n{f1_results_sorted[2][0]}: {f1_results_sorted[2][1]}\n\n")
    file.write(f"Best precision macro for label 0:\n{precision_results_sorted_0[0][0]}: {precision_results_sorted_0[0][1]} (Recall = {recall_results_sorted_0[0][1]})\n{precision_results_sorted_0[1][0]}: {precision_results_sorted_0[1][1]} (Recall = {recall_results_sorted_0[1][1]})\n{precision_results_sorted_0[2][0]}: {precision_results_sorted_0[2][1]} (Recall = {recall_results_sorted_0[2][1]})\n\n")
    file.write(f"Best precision macro for label 1:\n{precision_results_sorted_1[0][0]}: {precision_results_sorted_1[0][1]} (Recall = {recall_results_sorted_1[0][1]})\n{precision_results_sorted_1[1][0]}: {precision_results_sorted_1[1][1]} (Recall = {recall_results_sorted_1[1][1]})\n{precision_results_sorted_1[2][0]}: {precision_results_sorted_1[2][1]} (Recall = {recall_results_sorted_1[2][1]})\n\n")
    file.write(f"Best precision macro for label 2:\n{precision_results_sorted_2[0][0]}: {precision_results_sorted_2[0][1]} (Recall = {recall_results_sorted_2[0][1]})\n{precision_results_sorted_2[1][0]}: {precision_results_sorted_2[1][1]} (Recall = {recall_results_sorted_2[1][1]})\n{precision_results_sorted_2[2][0]}: {precision_results_sorted_2[2][1]} (Recall = {recall_results_sorted_2[2][1]})\n\n")
