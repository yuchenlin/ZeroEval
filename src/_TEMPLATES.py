import json

INSTRUCTION_MC_TEMPLATE_ZS = """
{question}

{choices}

Please answer this question by selecting the best choice. For example, if you believe the answer is "(C) candles", you should output "### Answer: (C) candles" at the end.
"""


INSTRUCTION_MC_TEMPLATE_ZS_COT = """
{question}

{choices}

Please answer this question by selecting the best choice. For example, if you believe the answer is "(C) candles", you should output "### Answer: (C) candles" at the end.

Let's think step by step, and then choose the best answer.
"""
 

ZEBRA_GRID_COT_TEMPLATE = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string

def apply_mc_template(item, cot=False):
    question, choices = item["question"], item["choices"]
    if cot.lower() == "true":
        return INSTRUCTION_MC_TEMPLATE_ZS_COT.format(question=question, choices=generate_choice_string(choices))
    elif cot.lower() == "false":
        return INSTRUCTION_MC_TEMPLATE_ZS.format(question=question, choices=generate_choice_string(choices))
    else:
        raise ValueError("Invalid value for cot. Please specify either 'true' or 'false'.")
    

def apply_lgp_grid_template(item, cot = "True"):
    if cot.lower() == "true":
        primpt_str = ZEBRA_GRID_COT_TEMPLATE[:]
    elif cot.lower() == "false":
        raise ValueError("COT=false is not supported for grid mode.")
    else:
        raise ValueError("Invalid value for cot. Please specify either 'true' or 'false'.")
    primpt_str = primpt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = primpt_str.replace("{json_template}", json_str)
    return prompt_str


if __name__ == "__main__":
    from datasets import load_dataset
    import random 

    dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    dataset = list(dataset)
    # shuffule 
    random.shuffle(dataset)
    for item in dataset:
        print(apply_lgp_grid_template(item, cot="True"))
        print("-"*100)
        print(json.dumps(item["solution"], indent=2))
        break



    