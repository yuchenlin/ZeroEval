import json

def escape_html(text):
    """Escape special characters in the given text for HTML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def format_text_with_newlines(text):
    if text is None:
        return ""
    return "<br>".join(escape_html(line) for line in text.split("\n"))

def write_html(task_summary, output_file):
    """
    Write the task summary to a beautiful HTML file.
    
    Highlights models that are correct and incorrect on each example.
    Only visualizes examples with correct_ratio < 10%.
    Includes the answer and respects newline characters in strings.
    Adds clickable model names to reveal reasoning.
    
    Args:
    task_summary (dict): Contains task information and model results
    output_file (str): Path to save the HTML file
    """
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Task Summary Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 10px; }
            .correct { background-color: #90EE90; }
            .incorrect { background-color: #FFB6C1; }
            table { border-collapse: collapse; width: 100%; font-size: 14px; }
            th, td { border: 1px solid #ddd; padding: 4px; text-align: left; }
            th { background-color: #f2f2f2; }
            .model-name { cursor: pointer; }
            .reasoning { display: none; margin-top: 5px; padding: 5px; background-color: #f9f9f9; border: 1px solid #ddd; font-size: 12px; }
            h1 { font-size: 24px; margin-bottom: 10px; }
            h2 { font-size: 18px; margin-top: 15px; margin-bottom: 5px; }
            p { margin: 5px 0; }
        </style>
        <script>
            function toggleReasoning(modelId) {
                var reasoning = document.getElementById(modelId);
                reasoning.style.display = reasoning.style.display === "none" ? "block" : "none";
            }
        </script>
    </head>
    <body>
        <h1>Task Summary Visualization</h1>
    """
    
    for example_id, example in task_summary.items():
        if example['correct_ratio'] < 0.1:  # Only process examples with correct_ratio < 10%
            html_content += f"""
            <h2>Example: {example_id}</h2>
            <p><strong>Question:</strong> {format_text_with_newlines(example['question'])}</p>
            <p><strong>Correct Answer:</strong> {format_text_with_newlines(example['correct_answer'])}</p>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Correct</th>
                    <th>Answer</th>
                </tr>
            """
            
            correct_models = example['correct_models'].split('|') if example['correct_models'] else []
            incorrect_models = example['incorrect_models'].split('|') if example['incorrect_models'] else []
            
            model_id = 0
            for model in correct_models + incorrect_models:
                answer = example['model_answers'].get(model, 'N/A')
                reasoning = example['reasoning'].get(model, 'No reasoning provided')
                is_correct = model in correct_models
                unique_model_id = f"model_{example_id}_{model_id}"  # Create a unique ID for each model
                html_content += f"""
                <tr class="{'correct' if is_correct else 'incorrect'}">
                    <td><span class="model-name" onclick="toggleReasoning('{unique_model_id}')">{escape_html(model)}</span></td>
                    <td>{'True' if is_correct else 'False'}</td>
                    <td>{format_text_with_newlines(answer)}</td>
                </tr>
                <tr>
                    <td colspan="3">
                        <div id="{unique_model_id}" class="reasoning">
                            <strong>Reasoning:</strong> {format_text_with_newlines(reasoning)}
                        </div>
                    </td>
                </tr>
                """
                model_id += 1
            
            html_content += "</table>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

# Read all JSON files in the current folder and convert to HTML
import os
import json

def process_task(task_name):
    json_file = f'state_of_limit/task_summary_{task_name}.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            task_summary = json.load(f)
        output_file = f'state_of_limit/output_{task_name}.html'
        write_html(task_summary, output_file)
        print(f"Generated HTML for {task_name}")
    else:
        print(f"JSON file for {task_name} not found")

# Process each task
for task in ['gsm', 'math-l5', 'crux', 'mmlu-redux']:
    process_task(task)



