import json 

# Load the prediction file
# filepath = "result_dirs/mmlu-redux/cot=True/gpt-3.5-turbo-0125.json"
# filepath = "result_dirs/mmlu-redux/cot=True/Magpie-Pro-SFT-v0.1.json"
# filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Llama-3-8B-Tulu-330K.json"
filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Magpie-Pro-SFT-v0.1.json"

filepath = filepath.replace("ZeroEval/", "")
with open(filepath, 'r') as f:
    predictions = json.load(f)

answers = {}
total = 0 
correct = 0
eval_ed_ids = set()

total_examples = 0
missing = 0
correct = 0 
for prediction_item in predictions: 
    total_examples += 1
    answer = prediction_item["correct_answer"]
    output = prediction_item["output"][0]
    if answer in output:
        correct += 1
    # if "### Answer: " not in output:
    #     print(output)
    #     missing += 1
    #     continue
    # if answer in output.split("##Answer: ")[1]: 
    #     correct += 1

print(f"File: {filepath}")
print(f"Total examples: {total_examples}")
print(f"Missing choices: {missing}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total_examples}")

exit()



example_prediction_item = """
{
    "session_id": "mmlu-redux-anatomy-#0",
    "chat_history": [
      "## Instruction \n\nYou will be presented with a question with multiple choices. \nYour task is to read the question carefully, consider the provided options, and select the one answer that you believe is the best choice. \n\n\n## Formatting \n\nPlease output both the letter and the answer text of the selected choice. \nFor example, if you believe the answer is \"(C) candles\", you should finally output \"### Answer: (C) candles\".\n\n## Question\n\nWhich of the following is a disorder characterized by uncontrollable episodes of falling asleep during the day?\n\n### Choices\n\n- (A) Dyslexia\n- (B) Epilepsy\n- (C) Hydrocephalus\n- (D) Narcolepsy\n\n"
    ],
    "model_input": "n/a",
    "output": [
      "### Answer: (D) Narcolepsy"
    ],
    "generator": "openai/gpt-3.5-turbo-0125",
    "configs": {
      "engine": "openai",
      "repetition_penalty": 1,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 4096
    },
    "dataset": "mmlu-redux",
    "id": "mmlu-redux-anatomy-#0",
    "question": "Which of the following is a disorder characterized by uncontrollable episodes of falling asleep during the day?",
    "choices": [
      "Dyslexia",
      "Epilepsy",
      "Hydrocephalus",
      "Narcolepsy"
    ],
    "correct_answer": "Narcolepsy",
    "source": "edinburgh-dawg/mmlu-redux",
    "config": "anatomy",
    "task_type": "multiple_choice"
  }
  
"""
