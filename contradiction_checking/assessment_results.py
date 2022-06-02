import os
import re

consultation_name = "repnum_with_titles"
assessment_results = {}

for model_name in os.listdir(f"../results/contradiction_checking/{consultation_name}"):
    if os.path.isdir(f"../results/contradiction_checking/{consultation_name}/{model_name}"):
        for file_name in os.listdir(f"../results/contradiction_checking/{consultation_name}/{model_name}"):
            if "metrics" in file_name:
                with open(f"../results/contradiction_checking/{consultation_name}/{model_name}/{file_name}", "r") as file:
                    if "contradictionshare" in file_name:
                        threshold = 0
                        for i, line in enumerate(file):
                            if i % 4 == 3:
                                threshold += 0.1
                                assessment_results[f"{model_name} {file_name} {str(threshold)}"] = float(re.findall("\d+\.\d+", line)[0])
                    else:
                        for i, line in enumerate(file):
                            if i == 2:
                                assessment_results[f"{model_name} {file_name}"] = float(re.findall("\d+\.\d+", line)[0])

results_sorted = sorted(assessment_results.items(), key=lambda x: x[1], reverse=True)

with open(f"../results/contradiction_checking/{consultation_name}/assessment_results.log", "w") as file:
    file.write(f"Best F1 macro:\n{results_sorted[0][0]}: {results_sorted[0][1]}\n{results_sorted[1][0]}: {results_sorted[1][1]}\n{results_sorted[2][0]}: {results_sorted[2][1]}")
