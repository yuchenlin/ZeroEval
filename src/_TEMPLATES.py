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

def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string

def apply_mc_template(question, choices, cot=False):
    if cot.lower() == "true":
        return INSTRUCTION_MC_TEMPLATE_ZS_COT.format(question=question, choices=generate_choice_string(choices))
    elif cot.lower() == "false":
        return INSTRUCTION_MC_TEMPLATE_ZS.format(question=question, choices=generate_choice_string(choices))
    else:
        raise ValueError("Invalid value for cot. Please specify either 'true' or 'false'.")