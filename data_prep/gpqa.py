import datasets
import json

dataset_path = "Idavidrein/gpqa"
dataset_name = "gpqa_diamond"
# dataset_name = "gpqa_main"

dataset = datasets.load_dataset(dataset_path, dataset_name, split="train")
import random
random.seed(0)

def shuffle_choices_and_create_example(row, index):
    list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
    random.shuffle(list_choices)
    new_example = {
        "id": f"{dataset_name}_{index}",
        "question": None,
        "choices": list_choices,
        "correct_answer": row['Correct Answer'],
        "source": dataset_path,
        "config": dataset_name,
        "task_type": "multiple_choice",
    }
    prompt = f"What is the correct answer to this question: {row['Question']}"
    prompt += f"\n\nChoices:\n(A) {list_choices[0]}\n(B) {list_choices[1]}\n(C) {list_choices[2]}\n(D) {list_choices[3]}"
    prompt += "\nAnswer with the letter of the correct choice."
    new_example["question"] = prompt
    return new_example

dataset = dataset.map(shuffle_choices_and_create_example, with_indices=True, remove_columns=dataset.column_names)
dataset.push_to_hub(
    repo_id="DongfuJiang/zeroeval",
    config_name=dataset_name,
    split='test',
    commit_message=f"Add {dataset_name} dataset",
)