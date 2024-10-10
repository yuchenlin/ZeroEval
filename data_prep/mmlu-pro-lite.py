import datasets
import json

dataset_path = "TIGER-Lab/MMLU-Pro"
dataset_name = "mmlu_pro_lite"


dataset = datasets.load_dataset(dataset_path, split="test")

total_sampled_num = 1000
dataset = dataset.shuffle(seed=0).select(range(total_sampled_num))

def shuffle_choices_and_create_example(row, index):
    new_example = {
        "id": f"{dataset_name}_{row['question_id']}",
        "question": None,
        "choices": row['options'],
        "correct_answer": row['options'][row['answer_index']],
        "source": dataset_path,
        "config": dataset_name,
        "task_type": "multiple_choice",
    }
    list_choices = row['options']
    prompt = f"What is the correct answer to this question: {row['question']}"
    # prompt += f"\n\nChoices:\n(A) {list_choices[0]}\n(B) {list_choices[1]}\n(C) {list_choices[2]}\n(D) {list_choices[3]}"
    prompt += "\n\nChoices:"
    for i, choice in enumerate(list_choices):
        prompt += f"\n({chr(65+i)}) {choice}"
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