import sys
import os
import time 
from functools import wraps
from typing import List 
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import cohere
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import Anthropic
 
from datasets import load_dataset
from tqdm import tqdm
from fastchat_conversation import map_to_conv
import json   

def apply_template(chat_history, model_name):
    model_inputs = [] 
    for chats in tqdm(chat_history, desc="Applying template", disable=True):
        if "gpt-" in model_name.lower():
            model_inputs.append("n/a") # gpt-s will be handled by another method.
            continue
        elif "gemini-" in model_name.lower():
            model_inputs.append("n/a") # gpt-s will be handled by another method.
            continue
        elif "cohere" in model_name.lower():
            model_inputs.append("n/a") # gpt-s will be handled by another method.
            continue
        elif "anthropic" in model_name.lower():
            model_inputs.append("n/a") # gpt-s will be handled by another method.
            continue
        else:
            conv = map_to_conv(model_name)
        for chat_id, chat in enumerate(chats):
            conv.append_message(conv.roles[chat_id%2], chat)
        conv.append_message(conv.roles[1], None)
        model_inputs.append(conv.get_prompt())
    return model_inputs

 
def load_eval_data(args, data_name=None, model_name=None):
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name    
    chat_history = []
    id_strs = []
    metadata = {}
    if data_name == "wild_bench":
        dataset = load_dataset("WildEval/WildBench", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        metadata = {"dataset": []}
    elif data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test") 
        metadata = {"dataset": [], "source_id": []}
    elif data_name == "mt-bench":
        dataset = load_dataset("json", data_files="https://huggingface.co/spaces/lmsys/mt-bench/raw/main/data/mt_bench/question.jsonl", split="train")
        metadata = {"question_id": [], "category": []}        
        if args.mt_turn == 2:
            with open(args.mt_turn1_result, "r") as f:
                mt_turn1_result = json.load(f)
            id_to_turn1_result = {}
            for item in mt_turn1_result:
                id_to_turn1_result[item["question_id"]] = item["turn1_output"] 
    else:
        raise ValueError(f"Data name {data_name} not supported")
    
    print(f"Loaded {len(dataset)} examples from {data_name}")

    for ind, item in enumerate(dataset):
        if data_name == "wild_bench":
            assert item["conversation_input"][-1]["role"] == "user"
            extracted_chats = [chat["content"] for chat in item["conversation_input"]]
            chat_history.append(extracted_chats)
            id_strs.append(item["session_id"])
        elif data_name in ["alpaca_eval", "just_eval"]:
            in_text = item["instruction"]    
            id_strs.append(item.get("id", str(ind)))
            chat_history.append([in_text])
        elif data_name == "mt-bench":
            if args.mt_turn == 1:
                chat_history.append([item["turns"][0]])
            elif args.mt_turn == 2:
                chat_history.append([item["turns"][0], 
                                     id_to_turn1_result[item["question_id"]], 
                                     item["turns"][1]]) 
            else:
                raise ValueError(f"mt_turn {args.mt_turn} not supported; must be 1 or 2")
        else:
            raise ValueError(f"Data name {data_name} not supported")
        for key in metadata: 
            assert key in item, f"Key {key} not found in metadata"
            metadata[key].append(item[key])
    print("Start applying template")
    model_inputs = apply_template(chat_history, model_name)
    return id_strs, chat_history, model_inputs, metadata



def clear_output(output, model_name): 
    """
    You can customize the output clearing logic here based on the model_name.
    """
    output = output.replace("<|endoftext|>", " ")
    output = output.strip()
    return output


def save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath):
    formatted_outputs = []
    if args.data_name == "wild_bench":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["session_id"] = id_strs[ind]
            output_item["chat_history"] = chat_history[ind]
            output_item["model_input"] = model_inputs[ind]
            output_item["output"] = [clear_output(o, args.model_name) for o in outputs[ind]]
            output_item["generator"] = args.model_name
            output_item["configs"] = {
                    "engine": args.engine,
                    "repetition_penalty": args.repetition_penalty,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                }
            output_item["dataset"] = args.data_name 
            for key in metadata:
                output_item[key] = metadata[key][ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "alpaca_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = [clear_output(outputs[ind][x].rstrip(), args.model_name) for x in range(len(outputs[ind]))]
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "just_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["id"] = ind
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["source_id"] = metadata["source_id"][ind]
            output_item["datasplit"] = "just_eval"
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "mt-bench":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["question_id"] = metadata["question_id"][ind]
            output_item["category"] = metadata["category"][ind]
            output_item[f"turn{args.mt_turn}_output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["model_id"] = args.model_name
            output_item["turn_id"] = args.mt_turn
            output_item["model_input"] = model_inputs[ind]
            output_item["configs"] = {
                "engine": args.engine,
                "repetition_penalty": args.repetition_penalty,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            formatted_outputs.append(output_item)
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)
        
 
def retry_handler(retry_limit=10):
    """
        This is an error handler for requests to OpenAI API.
        If will retry for the request for `retry_limit` times if the error is not a rate limit error.
        Otherwise, it will wait for the time specified in the error message and constantly retry.
        You can add specific processing logic for different types of errors here.

        Args:
            retry_limit (int, optional): The number of times to retry. Defaults to 3.
        
        Usage:
            @retry_handler(retry_limit=3)
            def call_openai_api():
                pass
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retried = 0
            flag_cohere_retry = False
            while True:
                try:
                    sys.stdout.flush()
                    if flag_cohere_retry:
                        kwargs['shorten_msg_times'] = retried
                    return func(*args, **kwargs)
                except Exception as e:
                    # if rate limit error, wait 2 seconds and retry
                    if isinstance(e, openai.error.RateLimitError):
                        words = str(e).split(' ')
                        try:
                            time_to_wait = int(words[words.index('after') + 1])
                        except ValueError:
                            time_to_wait = 5
                        # print("Rate limit error, waiting for {} seconds for another try..".format(time_to_wait))
                        time.sleep(time_to_wait) # wait 30 seconds
                        # print("Finished waiting for {} seconds. Start another try".format(time_to_wait))
                    elif isinstance(e, openai.error.APIError):
                        # this is because the prompt contains content that is filtered by OpenAI API
                        print("API error:", str(e))
                        if "Invalid" in str(e):
                            print("Invalid request, returning.")
                            raise e
                    else:
                        err_msg = str(e)
                        print(e.__class__.__name__+":", err_msg)
                        if retried < retry_limit:
                            if 'cohere' in e.__class__.__name__.lower() and 'prompt exceeds context length' in err_msg:
                                print ('cohere prompt length issue!')
                                flag_cohere_retry = True
                                return ['']
                                #raise e
                            print(f"Retrying for the {retried + 1} time..")
                            #if 'output blocked by content filtering policy' in err_msg.lower():
                            #    raise e
                        else:
                            # finally failed
                            print("Retry limit reached. Saving the error message and returning.")
                            print(kwargs["prompt"])
                            raise e
                        retried += 1
        return wrapper
    return decorate

def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
        # messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    
    response = openai.ChatCompletion.create(
        model=model,
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        **kwargs,
    )
    contents = []
    for choice in response['choices']:
        # Check if the response is valid
        if choice['finish_reason'] not in ['stop', 'length']:
            raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
        contents.append(choice['message']['content'])

    return contents
     
 
def google_chat_request(
    model: str=None,
    generation_config: dict=None,
    prompt: str=None,
    messages: List[dict]=None,
) -> List[str]:
    """
    Request the evaluation prompt from the Google API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"user","parts": ["You are an AI assistant that helps people find information."]},
                    {"role":"model", "parts": ["Understood."]},
                {"role":"user","parts": [prompt]}]

    #import pdb; pdb.set_trace()
    messages = [Content(role= message["role"], parts=[Part.from_text(part) for part in message["parts"]]) for message in messages]

    project_id = "grammarcorrection"
    location = "us-central1"
    vertexai.init(project=project_id, location=location)
    google_model = GenerativeModel(model)
    
    response = google_model.generate_content(
        messages,
        generation_config=generation_config,
    )
    #import pdb; pdb.set_trace()
    if len(response.candidates) == 0:
        output = '' # TODO: what should be done here?
        #import pdb; pdb.set_trace()
    #if len(response.candidates[0].content.parts) == 0:
    #    import pdb; pdb.set_trace()
    else:
        candidate = response.candidates[0]
        if candidate.finish_reason != 1 and candidate.finish_reason != 2:
            output = '' # TODO: what should be done here?
        else:
            output = candidate.content.parts[0].text
    contents = [output] #TODO: check stop reason? multiple candidates?

    return contents


def cohere_chat_request(
    model: str=None,
    engine: str=None,
    system_msg: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    shorten_msg_times: int=0,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"User","message": prompt}]
    #import pdb; pdb.set_trace()
    co = cohere.Client(os.getenv('COHERE_API_KEY'))
    assert messages[-1]['role'] == 'User', messages[-1]['role']
    #import pdb; pdb.set_trace()
    chat_history = messages[:-1]
    message = messages[-1]['message']
    for _ in range(shorten_msg_times):
        if len(chat_history) > 0:
            if _ == shorten_msg_times - 1:
                print ('removing past context')
            chat_history = chat_history[2:]
        else:
            msg_len = len(message)
            msg_len = msg_len // 2
            if _ == shorten_msg_times - 1:
                print (f'shorten msg len to {msg_len}')
            message = message[msg_len:]
    if len(chat_history) == 0:
        chat_history = None
    response = co.chat(
         message=message,
         preamble=system_msg,
         chat_history=None,
         model=model,
         temperature=temperature,
         p=top_p,
         max_tokens=max_tokens,
         prompt_truncation='AUTO') # TODO: frequency and presence penalty, stop
    return [response.text]


def mistral_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)
    #import pdb; pdb.set_trace()
    response = client.chat(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[ChatMessage(role=message['role'], content=message['content']) for message in messages],
    )
    #print(chat_response.choices[0].message.content)
    
    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    #import pdb; pdb.set_trace()

    return contents
     
def anthropic_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    system_msg: str=None,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        system_msg (str): The system prompt. 
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"user","content": prompt}]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        max_tokens=max_tokens,
        system=system_msg,
        messages=messages,
        stop_sequences=stop,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    
    contents = [response.content[0].text]
    return contents
