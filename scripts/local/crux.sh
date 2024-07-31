# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-3.5-turbo-0125 -p gpt-3.5-turbo-0125 -s 1
# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4o-mini-2024-07-18 -p gpt-4o-mini-2024-07-18 -s 8
# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4o-2024-05-13 -p gpt-4o-2024-05-13 -s 8

# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4-0314 -p gpt-4-0314 -s 8
# bash zero_eval_api.sh -d crux -f openai -m openai/gpt-4-turbo-2024-04-09 -p gpt-4-turbo-2024-04-09 -s 8 

bash zero_eval_api.sh -d crux -f anthropic -m anthropic/claude-3-5-sonnet-20240620 -p claude-3-5-sonnet-20240620 -s 8 
bash zero_eval_api.sh -d crux -f anthropic -m anthropic/claude-3-haiku-20240307 -p claude-3-haiku-20240307 -s 8
bash zero_eval_api.sh -d crux -f anthropic -m anthropic/claude-3-opus-20240229 -p claude-3-opus-20240229 -s 8
bash zero_eval_api.sh -d crux -f anthropic -m anthropic/claude-3-sonnet-20240229 -p claude-3-sonnet-20240229 -s 8


bash zero_eval_api.sh -d crux -f together -m google/gemma-2-9b-it -p gemma-2-9b-it -s 8
bash zero_eval_api.sh -d crux -f together -m google/gemma-2-27b-it -p gemma-2-27b-it -s 8

bash zero_eval_api.sh -d crux -f together -m meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo -p Llama-3.1-405B-Inst-fp8 -s 8 -x 2048

bash zero_eval_api.sh -d crux -f google -m google/gemini-1.5-pro -p gemini-1.5-pro -s 8
bash zero_eval_api.sh -d crux -f google -m google/gemini-1.5-flash -p gemini-1.5-flash -s 8



bash zero_eval_api.sh -d crux -f openai -m deepseek-chat -p deepseek-v2-chat-0628 -s 8
bash zero_eval_api.sh -d crux -f openai -m deepseek-coder -p deepseek-v2-coder-0724 -s 8


bash zero_eval_api.sh -d crux -f reka -m reka/reka-flash-20240226 -p reka-flash-20240226 -s 8
bash zero_eval_api.sh -d crux -f reka -m reka/reka-core-20240501 -p reka-core-20240501 -s 8

bash zero_eval_api.sh -d crux -f openai -m yi-large -p yi-large -s 8
bash zero_eval_api.sh -d crux -f openai -m yi-large-preview -p yi-large-preview -s 8


bash zero_eval_api.sh -f mistral -d crux -m mistral-large-2407 -p Mistral-Large-2 -s 8