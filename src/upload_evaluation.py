import json 
from datasets import load_dataset, Dataset
import os 
import sys

def load_and_upload(model_name, evaluator="gpt-4-0125-preview", reference=None, with_checklist=True):
    if reference == "arena":
        filepath = f"evaluation/results/eval={evaluator}/arena/{model_name}.json"
        config_name = "arena-" + model_name+"-eval="+evaluator 
    else:
        if with_checklist:
            filepath = f"evaluation/results/eval={evaluator}/ref={reference}/{model_name}.json"
            config_name = model_name+"-eval="+evaluator+"-ref="+reference
        else:
            filepath = f"evaluation/results/eval={evaluator}_woCL/ref={reference}/{model_name}.json"
            config_name = model_name+"-eval="+evaluator + '_woCL' + "-ref="+reference
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {model_name} results with {len(data)} samples.")
    if len(data) != 1024:
        print(f"Expected 1024 samples, got {len(data)}. Exit!")
        return
    dataset = Dataset.from_list(data)
    
    dataset.push_to_hub(
        repo_id="WildEval/WildBench-Evaluation",
        config_name=config_name,
        split=f"train",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {model_name} results. Evaluated with {evaluator} and referenced with {reference}.",
    )
    print(f"Uploaded {model_name} results.")


if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python src/upload_evaluation.py <evaluator> <reference> <model_name> <with_checklist>")
        sys.exit(1)
    if sys.argv[2] == "arena":
        round_name = sys.argv[3]
        load_and_upload(model_name=round_name, evaluator=sys.argv[1], reference="arena")
    else:
        load_and_upload(model_name=sys.argv[3], evaluator=sys.argv[1], reference=sys.argv[2], with_checklist=sys.argv[4]=='True')

"""
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gpt-4-0125-preview
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 tulu-2-dpo-70b
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mistral-7B-Instruct-v0.2
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mixtral-8x7B-Instruct-v0.1
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Yi-34B-Chat
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 vicuna-13b-v1.5

python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-70b-chat-hf
# python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-13b-chat-hf
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-7b-chat-hf

python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mistral-7B-Instruct-v0.1
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gemma-7b-it
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gemma-2b-it

python src/upload_evaluation.py gpt-4-0125-preview arena round1
"""
