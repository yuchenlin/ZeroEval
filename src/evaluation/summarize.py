import pandas as pd

# Load the JSON files
gsm_file = 'result_dirs/gsm.summary.json'
mmlu_file = 'result_dirs/mmlu-redux.summary.json'
zebra_file = 'result_dirs/zebra-grid.summary.json'

gsm_data = pd.read_json(gsm_file)
mmlu_data = pd.read_json(mmlu_file)
zebra_data = pd.read_json(zebra_file)

# replace the value from "gemma-2-9b-it@nvidia" to "gemma-2-9b-it" for all data. only when the model name is "gemma-2-9b-it@nvidia"
def replace_model_names(cur_data):
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-9b-it@nvidia', 'gemma-2-9b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-9b-it@together', 'gemma-2-9b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-27b-it@together', 'gemma-2-27b-it') 
    cur_data['Model'] = cur_data['Model'].replace('gemma-2-27b-it@nvidia', 'gemma-2-27b-it') 
    cur_data['Model'] = cur_data['Model'].replace('deepseek-chat', 'deepseek-v2-chat-0628')
    cur_data['Model'] = cur_data['Model'].replace('deepseek-coder', 'deepseek-v2-coder-0614')
    return cur_data

gsm_data = replace_model_names(gsm_data)
mmlu_data = replace_model_names(mmlu_data)
zebra_data = replace_model_names(zebra_data)


# Add suffixes to the columns
gsm_data = gsm_data[['Model', 'Acc']]
gsm_data = gsm_data.add_suffix('_gsm')
gsm_data = gsm_data.rename(columns={'Model_gsm': 'Model'}).rename(columns={'Acc_gsm': 'GSM'})

mmlu_data = mmlu_data[['Model', 'Acc']]
mmlu_data = mmlu_data.add_suffix('_mmlu')
mmlu_data = mmlu_data.rename(columns={'Model_mmlu': 'Model'}).rename(columns={'Acc_mmlu': 'MMLU<br/>-Redux'})

# remove the rows when the mode = sampling 
zebra_data = zebra_data[zebra_data['Mode'] != 'sampling']
zebra_data = zebra_data[['Model', "Easy Puzzle Acc"]] #  'Puzzle Acc', 
zebra_data = zebra_data.add_suffix('_zebra')
zebra_data = zebra_data.rename(columns={'Model_zebra': 'Model'}).rename(columns={'Puzzle Acc_zebra': 'ZebraLogic<br/>-Full'}).rename(columns={'Easy Puzzle Acc_zebra': 'ZebraLogic<br/>-Easy'})



# Merge the dataframes on the "Model" column
merged_data = pd.merge(gsm_data, mmlu_data, on='Model')
merged_data = pd.merge(merged_data, zebra_data, on='Model')


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