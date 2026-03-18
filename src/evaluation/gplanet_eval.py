import json
import re
from tabulate import tabulate
import os
import sys
from eval_utils import (
    load_model_results,
    extract_values_from_json,
    extract_first_complete_json,
    model_specific_extraction,
    model_name_replacement,
)


def rank_elements(list_elements):
    # Rank elements while handling ties
    sorted_elements = sorted(list_elements)
    ranks = [sorted_elements.index(x) + 1 for x in list_elements]
    return ranks


def spearmanr_manual(list1, list2):
    # Get ranks of both lists
    rank1 = rank_elements(list1)
    rank2 = rank_elements(list2)

    # Calculate the difference in ranks and square it
    n = len(list1)
    d_squared = sum((rank1[i] - rank2[i]) ** 2 for i in range(n))

    # Apply the Spearman rank correlation formula
    spearman_corr = 1 - (6 * d_squared) / (n * (n**2 - 1))

    return spearman_corr


def calculate_similarity(list1, list2):
    common_elements = [element for element in list1 if element in list2]
    if not common_elements:
        return 0
    if len(common_elements) == 1:
        return 0.5
    indices1 = [list1.index(element) for element in common_elements]
    indices2 = [list2.index(element) for element in common_elements]
    correlation = spearmanr_manual(indices1, indices2)
    return (correlation + 1) / 2


def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def extract_output_and_truth(data, action_pattern=None):
    output_plans = []
    truth_plans = []
    for item in data:
        try:
            # action = item["output"][0].split("###")[-1]
            # we should extract the action from the output
            action = item["output"][0].split("<final_action_sequence>")[1].split("</final_action_sequence>")[0]
        except:
            action = item["output"][0]
        output_plan = re.findall(action_pattern, action) if action_pattern else [action]
        truth_plan = item["truth_labels"]
        output_plans.append(output_plan)
        truth_plans.append(truth_plan)
    return output_plans, truth_plans


def eval_model(model, filepath):
    data = load_data(filepath)
    action_pattern = r"\([A-Z]\)"
    output_plans, truth_plans = extract_output_and_truth(data, action_pattern)

    solved_examples, no_answer = 0, 0
    precisions, recalls, fs, rs, ps = [], [], [], [], []
    parsed_results = []
    num_total_examples = len(data)
    for item, output, truth in zip(data, output_plans, truth_plans):
        tp = len(set(output).intersection(set(truth)))
        recall = (tp + 1e-10) / (len(truth) + 1e-5)
        precision = (tp + 1e-10) / (len(output) + 1e-5)
        precisions.append(precision)
        recalls.append(recall)

        correct = output == truth
        if correct:
            solved_examples += 1

        if tp > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        fs.append(f1)
        rank_similarity = calculate_similarity(output, truth)
        rs.append(rank_similarity)
        plan_score = rank_similarity * recall
        ps.append(plan_score)
        parsed_item = item.copy()
        parsed_item["output"] = output
        parsed_item["truth"] = truth
        parsed_item["f1"] = f1
        parsed_item["recall"] = recall
        parsed_item["precision"] = precision
        parsed_item["matched"] = correct
        parsed_item["rank similarity"] = rank_similarity
        parsed_item["plan score"] = plan_score
        # parsed_item["correct_answer"] = {"raw": truth, "sanitized": truth}
        # parsed_item["matched"] = correct
        parsed_results.append(parsed_item)
    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Total"] = num_total_examples
    result["precision"] = f"{sum(precisions) / len(precisions) * 100:.2f}"
    result["recall"] = f"{sum(recalls) / len(recalls) * 100:.2f}"
    result["f1"] = f"{sum(fs) / len(fs) * 100:.2f}"
    result["average rank similarity"] = f"{sum(rs) / len(rs) * 100:.2f}"
    result["exact match"] = f"{solved_examples / len(data) * 100:.2f}"
    result["plan relative"] = f"{sum(ps) / len(ps) * 100:.2f}"
    return result, parsed_results


def gen_results(run_name_folders):
    model_results = load_model_results(run_name_folders)

    columns = [
        "Model",
        "Mode",
        "Total",
        "precision",
        "recall",
        "f1",
        "average rank similarity",
        "exact match",
        "plan relative",
    ]
    rows = []
    for model_name, filepath in model_results.items():
        result, parsed_results = eval_model(model_name, filepath)
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    rows = sorted(rows, key=lambda x: -float(x["plan relative"]))
    table_data = [[row[col] for col in columns] for row in rows]

    print(
        tabulate(
            table_data,
            headers=columns,
            tablefmt="fancy_outline",
            stralign="center",
            numalign="center",
        )
    )

    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/4666e72d-4202-4283-8e78-e5ce2b030dcf" alt="zebra_banner" style="width: 69%;" />
</div>
"""
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(
            banner_header
            + tabulate(
                table_data,
                headers=columns,
                tablefmt="github",
                stralign="center",
                numalign="center",
            )
        )

    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    
    data_name = "gplanet"  # by default if there is no sys.argv[1]
    if len(sys.argv) > 1:
        data_name = sys.argv[1]
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}",
        "sampling": f"result_dirs/{data_name}/sampling",
    }
    gen_results(run_name_folders)
