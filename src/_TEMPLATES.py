import json


from .templates.ZEBRA_GRID import ZEBRA_GRID
from .templates.MCQA import MCQA
from .templates.OEQA import OEQA, OEQA_DIRECT
from .templates.GPLANET import GPLANET

def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string


def apply_mc_template(item):
    question, choices = item["question"], item["choices"]
    prompt_str = MCQA[:]
    prompt_str = prompt_str.replace("{question}", question)
    prompt_str = prompt_str.replace("{choices}", generate_choice_string(choices))
    return prompt_str

def apply_oeqa_template(item, question_key="question", cot=True):
    question = item[question_key]
    if cot:
        prompt_str = OEQA[:]
    else:
        prompt_str = OEQA_DIRECT[:]
    prompt_str = prompt_str.replace("{question}", question)
    return prompt_str

def apply_lgp_grid_template(item):
    prompt_str = ZEBRA_GRID[:]
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str

def apply_gplanet_template(item):
    prompt_str = GPLANET[:]
    # use the markdown list of actions -- each item is in the format of "- (A) action"
    actions = '\n'.join([f"- ({chr(65 + i)}) {action}" for i, action in enumerate(item['shuffle_actions'])])

    # reformat item["objects_str"] --> strip the key names
    objects_str = item["objects_str"].replace("```json", "").replace("```", "")
    new_objects_str = []
    for dict in json.loads(objects_str):
        new_dict = {}
        for key, value in dict.items():
            new_dict[key.strip()] = value.strip()

        new_dict["position_xyz"] = f"({new_dict['position_x']}, {new_dict['position_y']}, {new_dict['position_z']})"
        new_dict["rotation_xyz"] = f"({new_dict['rotation_x']}, {new_dict['rotation_y']}, {new_dict['rotation_z']})"
        # remove the position_x, position_y, position_z, rotation_x, rotation_y, rotation_z
        new_dict.pop("position_x")
        new_dict.pop("position_y")
        new_dict.pop("position_z")
        new_dict.pop("rotation_x")
        new_dict.pop("rotation_y")
        new_dict.pop("rotation_z")
        if new_dict["position_xyz"] == "(, , )":
            new_dict["position_xyz"] = "N/A"
        if new_dict["rotation_xyz"] == "(, , )":
            new_dict["rotation_xyz"] = "N/A"
        if new_dict["parent_receptacle"] == "":
            new_dict["parent_receptacle"] = "N/A"
        new_objects_str.append(new_dict)
    objects_str = json.dumps(new_objects_str, indent=2)
    prompt_str = prompt_str.replace("{actions}", actions)
    prompt_str = prompt_str.replace("{objects_str}", objects_str)
    prompt_str = prompt_str.replace("{task}", item["task"])
    return prompt_str

if __name__ == "__main__":
    from datasets import load_dataset
    import random

    def mcqa_test():
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_mc_template(item))
            print("-"*100)
            break

    def zebra_test():
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_lgp_grid_template(item, cot="True"))
            print("-"*100)
            print(json.dumps(item["solution"], indent=2))
            break

    def gsm_test():
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item))
            print("-"*100)
            break

    def crux_test():
        dataset = load_dataset("flydust/zero-eval", "crux", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item, cot=True))
            print("-"*100)
            break

    def math_l5_test():
        dataset = load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item, question_key="problem"))
            print(item)
            print("-"*100)
            break

    def gplanet_test():
        dataset = load_dataset("WildEval/G-PlanET", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_gplanet_template(item))
            print(item.keys())
            print(item["truth_labels"])
            print("-"*100)
            break
    # mcqa_test()
    # gsm_test()
    # crux_test()
    # math_l5_test()
    gplanet_test()