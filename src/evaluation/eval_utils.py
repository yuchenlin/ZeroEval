import re 
import os 
import json 


def load_model_results(run_name_folders):
    model_results = {}
    for run_name, folder in run_name_folders.items():
        # iterate all json files under the folder 
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not filename.endswith(".json"):
                continue
            model_name = filename.replace(".json", "")  
            model_name = f"{model_name}%{run_name}"
            model_results[model_name] = filepath  
    return model_results

def extract_values_from_json(json_string, keys = ["reasoning", "answer"]):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
    return extracted_values



 
def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    
    return None


if __name__ == "__main__":
    json_test = """
    {
        "reasoning": "Calculate shipping cost ($1.40 per pound x 4 pounds) and mileage cost ($0.08 per mile x 20 miles), then add them together ($3.00). Determine refund amount (75% of $32 = $24) and subtract it from the total shipping cost to find Milly’s loss (-$21).",
        "answer": -21
    }
    """
    print(json.dumps(extract_last_complete_json(json_test), indent=2))
