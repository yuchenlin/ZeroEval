import pandas as pd

# Load the JSON files
gsm_file = 'result_dirs/gsm.summary.json'
mmlu_file = 'result_dirs/mmlu-redux.summary.json'
zebra_file = 'result_dirs/zebra-grid.summary.json'
crux_file = 'result_dirs/crux.summary.json'

gsm_data = pd.read_json(gsm_file)
mmlu_data = pd.read_json(mmlu_file)
zebra_data = pd.read_json(zebra_file)
crux_data = pd.read_json(crux_file)

# replace the value from "gemma-2-9b-it@nvidia" to "gemma-2-9b-it" for all data. only when the model name is "gemma-2-9b-it@nvidia"
def replace_model_names(cur_data):
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-9b-it@nvidia', 'gemma-2-9b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-9b-it@together', 'gemma-2-9b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-27b-it@together', 'gemma-2-27b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-27b-it@nvidia', 'gemma-2-27b-it') 
    cur_data['Model'] = cur_data['Model'].replace('deepseek-chat', 'deepseek-v2-chat-0628')
    cur_data['Model'] = cur_data['Model'].replace('deepseek-coder', 'deepseek-v2-coder-0614')
    cur_data['Model'] = cur_data['Model'].replace('DeepSeek-Coder-V2-0724', 'deepseek-v2-coder-0724')
    cur_data['Model'] = cur_data['Model'].replace('Llama-3.1-405B-Instruct-Turbo', 'Llama-3.1-405B-Inst-fp8')
    
    return cur_data

gsm_data = replace_model_names(gsm_data)
mmlu_data = replace_model_names(mmlu_data)
zebra_data = replace_model_names(zebra_data)


# Add suffixes to the columns
gsm_data = gsm_data[['Model', 'Acc']]
gsm_data = gsm_data.add_suffix('_gsm')
gsm_data = gsm_data.rename(columns={'Model_gsm': 'Model'}).rename(columns={'Acc_gsm': 'GSM8K'})

mmlu_data = mmlu_data[['Model', 'Acc']]
mmlu_data = mmlu_data.add_suffix('_mmlu')
mmlu_data = mmlu_data.rename(columns={'Model_mmlu': 'Model'}).rename(columns={'Acc_mmlu': 'MMLU<br/>-Redux'})

# remove the rows when the mode = sampling 
zebra_data = zebra_data[zebra_data['Mode'] != 'sampling']
zebra_data = zebra_data[['Model', "Puzzle Acc"]] #, 'Puzzle Acc']] #  'Puzzle Acc',  # Easy 
zebra_data = zebra_data.add_suffix('_zebra')
zebra_data = zebra_data.rename(columns={'Model_zebra': 'Model'}).rename(columns={'Puzzle Acc_zebra': 'ZebraLogic<br/>-Full'}).rename(columns={'Easy Puzzle Acc_zebra': 'ZebraLogic<br/>-Easy'})


crux_data = crux_data[['Model', 'Acc']]
crux_data = crux_data.add_suffix('_crux')
crux_data = crux_data.rename(columns={'Model_crux': 'Model'}).rename(columns={'Acc_crux': 'CRUX'})

# Merge the dataframes on the "Model" column
merged_data = pd.merge(gsm_data, mmlu_data, on='Model')
merged_data = pd.merge(merged_data, zebra_data, on='Model')
merged_data = pd.merge(merged_data, crux_data, on='Model')


# add a final column to do average of the scores except for Model name 
merged_data['Average'] = merged_data.drop(columns=['Model']).mean(axis=1)

# rank the models based on the average score
merged_data = merged_data.sort_values(by='Average', ascending=False)

# Generate a Markdown table
markdown_table = merged_data.to_markdown(index=False, floatfmt=".2f")

# rename the <br> in column names with "\n"
merged_data.columns = merged_data.columns.str.replace('<br/>', '\n')
markdown_table_lite = merged_data.to_markdown(index=False, floatfmt=".2f", tablefmt="grid")
print(markdown_table_lite)

# print(markdown_table)
with open('result_dirs/summary.md', 'w') as f:
    f.write(markdown_table)

# save the json output

with open('result_dirs/summary.json', 'w') as f:
    f.write(merged_data.to_json(orient='records', lines=False))