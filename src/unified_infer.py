from typing import List, Dict

from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer


from src.unified_utils import load_eval_data, save_outputs, prepare_save_outputs
from src.global_configs import HF_TEMPLATED_MODELS, IM_END_MODELS
from src.unified_utils import openai_chat_request, retry_handler, google_chat_request, cohere_chat_request, mistral_chat_request, anthropic_chat_request, together_chat_request, reka_chat_request
from src.hf_models import DecoderOnlyModelManager

from src.llm_engines import(
    create_vllm_async_engine, 
    run_vllm_async_inference,
    shutdown_vllm_async_engine
)
from src.config_parser import parse_args
from src.config_utils import get_shards_split



def infer_maybe_lora(model_name):
    if os.path.exists(model_name):
        if os.path.exists(f"{model_name}/adapter_config.json"):
            adapter_config_path = f"{model_name}/adapter_config.json"
            adapter_path = model_name
            lora_model = True
        else:
            lora_model = False
    else:
        # try hugging face
        from huggingface_hub import hf_hub_download, snapshot_download
        try:
            adapter_config_path = hf_hub_download(repo_id=model_name, filename="adapter_config.json")
            adapter_path = snapshot_download(repo_id=model_name)
            lora_model = True
        except Exception as e:
            lora_model = False
    if lora_model:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name_or_path = adapter_config["base_model_name_or_path"]
        lora_model = adapter_path
    else:
        base_model_name_or_path = model_name
        lora_model = None
    return base_model_name_or_path, lora_model


def sort_given_ids_order(samples: list, ids: List[str], ids_ranks: Dict[str,int]):
    """ Sort array of 'samples' using provided map (id: str -> rank: int)

        Each sample from the data has corresponding 'unique_id' in 'ids'

    Args:
        data (list): any array, length same as 'ids'
        ids (List[str]): array with strings each is a unique id
        ids_ranks (Dict[str,int]): maps 'unique_id' to an integer value

    """
    assert len(samples) == len(ids)

    idx_sort = sorted(range(len(ids)), key=lambda i: ids_ranks.get(ids[i], float('inf')))
    return [samples[i] for i in idx_sort]



if __name__ == "__main__":
    args = parse_args()

    print("loading dataset!")
    if args.use_hf_conv_template:
        HF_TEMPLATED_MODELS.append(args.model_name)
    if args.use_imend_stop:
        IM_END_MODELS.append(args.model_name)
    
    # TODO: we need to support the case when you have an existing file
    
    # Data loading
    id_strs, chat_history, model_inputs, metadata = load_eval_data(args)
    print("loading dataset ... done!")


    # Load the model
    print("loading model!")
    if args.tokenizer_name == "auto":
        args.tokenizer_name = args.model_name
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        max_model_len = None if args.max_model_len == -1 else args.max_model_len
        base_model_name_or_path, lora_model_name_or_path = infer_maybe_lora(args.model_name)
        if lora_model_name_or_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(lora_model_name_or_path.split("/")[-1], 1, lora_model_name_or_path)
        else:
            lora_request = None

        llm = LLM(model=base_model_name_or_path, tokenizer=args.tokenizer_name, tensor_parallel_size=args.tensor_parallel_size,
                        download_dir=args.download_dir, dtype=args.dtype, tokenizer_mode=args.tokenizer_mode,
                        max_model_len=max_model_len, trust_remote_code=True,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        enable_lora=lora_request is not None
                        )
    elif args.engine == "vllm_async":
        from vllm import SamplingParams   
        llm = create_vllm_async_engine(args)
        # llm = None
    elif args.engine == "hf":
        llm = DecoderOnlyModelManager(args.model_name, args.model_name, cache_dir=args.download_dir,
                                    bf16=args.hf_bf16, gptq=args.hf_gptq)
        llm.load_model()
    elif args.engine == "openai":
        pass
    elif args.engine == "google":
        pass
    elif args.engine == "cohere":
        pass
    elif args.engine == "anthropic":
        pass
    elif args.engine == "together":
        pass
    elif args.engine == "reka":
        pass



    # decide start_index and end_index by num_shards and shard_id
    full_data_size = len(id_strs)
    if args.num_shards>1:        
        shard_splits = get_shards_split(full_data_size, args.num_shards)
        args.start_index, args.end_index = shard_splits[args.shard_id]
    else:
        args.start_index, args.end_index = 0, full_data_size

    # Slice the data
    model_inputs = model_inputs[args.start_index:args.end_index]
    id_strs = id_strs[args.start_index:args.end_index]
    chat_history = chat_history[args.start_index:args.end_index]
    metadata = {key: metadata[key][args.start_index:args.end_index] for key in metadata}

    
    # Decide the output filepath
    if args.filepath == "auto":
        # Decide the output filepath
        if "/" in args.model_name and args.model_pretty_name is None:
            args.model_pretty_name = args.model_name.split("/")[-1]
        os.system(f"mkdir -p {args.output_folder}")
        if args.start_index == 0 and args.end_index == full_data_size:
            filepath = os.path.join(args.output_folder,f"{args.model_pretty_name}.json")
        else:
            filepath = os.path.join(args.output_folder,f"{args.model_pretty_name}.{args.start_index}-{args.end_index}.json")
    else:
        filepath = args.filepath
        output_folder = "/".join(filepath.split("/")[:-1])
        if not os.path.exists(output_folder):
            os.system(f"mkdir -p {output_folder}")


    # speical handling
    stop_words = []
    include_stop_str_in_output = False
    stop_token_ids = []

    if args.model_name in IM_END_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        potential_end_tokens = ["<|im_end|>" , "<|eot_id|>"]
        for potential_end_token in potential_end_tokens:
            if potential_end_token in hf_tokenizer.get_vocab():
                stop_token_ids += [hf_tokenizer.get_vocab()[potential_end_token]]
    if args.model_name in HF_TEMPLATED_MODELS:
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        stop_token_ids.append(hf_tokenizer.eos_token_id)


    outputs = []
    # Load the existing outputs
    # ideally we want to have first K prompts generated by LLM and saved in the file, then we can skip first K examples. 
    if os.path.exists(filepath) and not args.overwrite:
        with open(filepath) as f:
            formatted_outputs = json.load(f)
            formatted_outputs = formatted_outputs

        # we want to remove from shard data examples that were processed earlier
        shard_keys = set(id_strs)

        # place examples that were processed earlier in the beginning of the arrays
        sort_order = {item['session_id']: i for i, item in enumerate(formatted_outputs)}
        model_inputs = sort_given_ids_order(model_inputs, id_strs, sort_order)
        chat_history = sort_given_ids_order(chat_history, id_strs, sort_order)
        for key in metadata:
            metadata[key] = sort_given_ids_order(metadata[key], id_strs, sort_order)

        id_strs = sort_given_ids_order(id_strs, id_strs, sort_order)

        for i, output_item in enumerate(formatted_outputs):
            if output_item['session_id'] not in shard_keys:
                raise ValueError("Items that were saved as outputs in previous runs doesn't exist the dataset!")
            
            # this should happen after sorting 'id_strs'
            assert output_item['session_id'] == id_strs[i], f"session_id mismatch: {output_item['session_id']} != {id_strs[i]}"

            outputs.append([output_item["output"]] if type(output_item["output"]) == str else output_item["output"])
            if args.model_name.startswith("openai/o"):
                if "hidden_reasoning_token" not in metadata:
                    metadata["hidden_reasoning_token"] = []
                metadata["hidden_reasoning_token"].append(output_item["hidden_reasoning_token"])

    
    num_skipped = len(outputs)
    print(f"We skipped the first {num_skipped} examples")


    # Load the existing data from the cache_filepath
    cache_outputs = {}
    if args.cache_filepath is not None:
        if os.path.exists(args.cache_filepath):
            with open(args.cache_filepath) as f:
                cache_data = json.load(f)
            for output_item in cache_data:
                # if output_item["output"]  is a list and the first string is not empty
                if type(output_item["output"]) == list and len(output_item["output"]) > 0 and len(output_item["output"][0]) > 0:
                    cache_outputs[output_item["session_id"]] = output_item

        print(f"Loaded {len(cache_outputs)} non-empty outputs from the cache file: {args.cache_filepath}")

    todo_inputs = model_inputs[num_skipped:]


    print(f"Outputs will be saved in {filepath}")
    save_outputs_short = prepare_save_outputs(
                    args = args, 
                    id_strs = id_strs, 
                    chat_history = chat_history, 
                    metadata = metadata, 
                    model_inputs = model_inputs, 
                    filepath = filepath
                )

    if args.engine == "vllm":
        sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature,            
                                         repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens,
                                         stop=stop_words, stop_token_ids=stop_token_ids, include_stop_str_in_output=include_stop_str_in_output, n=args.num_outputs)
        
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False, lora_request=lora_request)
            outputs.extend([[o.text for o in x.outputs] for x in batch_outputs]) # TODO: enbale multiple generation
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
        save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "vllm_async":
        sampling_params = SamplingParams(top_p=args.top_p, 
                                         temperature=args.temperature,            
                                         repetition_penalty=args.repetition_penalty, 
                                         max_tokens=args.max_tokens,
                                         stop=stop_words, 
                                         stop_token_ids=stop_token_ids, 
                                         include_stop_str_in_output=include_stop_str_in_output, 
                                         n=args.num_outputs)
        new_outputs = run_vllm_async_inference(llm, args, sampling_params, todo_inputs, saver = save_outputs_short)
        outputs.extend(new_outputs)
        save_outputs_short(outputs = outputs)
        shutdown_vllm_async_engine(llm)

    elif args.engine == "hf":
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            sampling_params = {
                "do_sample": True if args.temperature > 0 else False,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "repitition_penalty": args.repetition_penalty,
                "eof_strings": "|".join(stop_words),
                "max_output_tokens": args.max_tokens,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            batch_outputs = llm.infer_generate(batch_inputs, args=sampling_params)
            outputs.extend(batch_outputs) # TODO: enbale multiple generation
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
        save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "openai":
        todo_chats = chat_history[num_skipped:]
        todo_ids = id_strs[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = openai_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            current_id_str = todo_ids[cur_id]
            # check if in the cache
            if current_id_str in cache_outputs:
                print(f"Using cache from {args.cache_filepath} for {current_id_str}")
                cache_item = cache_outputs[current_id_str]
                outputs.append(cache_item["output"])
                if "hidden_reasoning_token" not in metadata:
                    metadata["hidden_reasoning_token"] = []
                metadata["hidden_reasoning_token"].append(cache_item["hidden_reasoning_token"])
            else:
                openai_msg = [{"role":"system", "content":"You are a helpful AI assistant."}]
                for i, chat_item in enumerate(chat):
                    if i % 2 == 0:
                        openai_msg.append({"role":"user","content": chat_item})
                    else:
                        openai_msg.append({"role":"assistant","content": chat_item})
                openai_args = {
                    "model": args.model_name,
                    "prompt": None,
                    "messages": openai_msg,
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stop": stop_words,
                    "n": args.num_outputs  # Pass the num_outputs argument here
                }
                result = api(**openai_args)
                # for o1 and o3
                if args.model_name.startswith("openai/o"):
                    try:
                        content, hidden_reasoning_token = result
                    except Exception as e:
                        print(f"Error: {e}")
                        content = result
                        hidden_reasoning_token = 0
                    # print(f"hidden_reasoning_token: {hidden_reasoning_token}")
                    if "hidden_reasoning_token" not in metadata:
                        metadata["hidden_reasoning_token"] = []
                    metadata["hidden_reasoning_token"].append(hidden_reasoning_token)
                else:
                    content = result
                outputs.append(content)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "together":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = together_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            msgs = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    msgs.append({"role":"user","content": chat_item})
                else:
                    msgs.append({"role":"assistant","content": chat_item})
            openai_args = {
                "model": args.model_name.replace("@together", ""),
                "prompt": None,
                "messages": msgs,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**openai_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)


    elif args.engine == "google":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = google_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            #google_msg = [{"role":"user", "parts": ["You are a helpful AI assistant."]}]
            #google_msg.append({"role":"model", "parts": ["Understood."]})
            google_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    google_msg.append({"role":"user","parts": [chat_item,]})
                else:
                    google_msg.append({"role":"model","parts": [chat_item,]})
            google_args = {
                "model": args.model_name.replace("google/", ""),
                "messages": google_msg,
                'generation_config': {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_output_tokens": args.max_tokens,
                    "stop_sequences": stop_words,
                }
            }
            result = api(**google_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "cohere":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = cohere_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            system_msg = "You are a helpful AI assistant."
            cohere_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    cohere_msg.append({"role":"User","message": chat_item})
                else:
                    cohere_msg.append({"role":"Chatbot","message": chat_item})
            cohere_args = {
                "model": args.model_name,
                "prompt": None,
                "system_msg": system_msg,
                "messages": cohere_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            result = api(**cohere_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "mistral":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = mistral_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            mistral_msg = [{"role":"system", "content":"You are a helpful AI assistant."}]
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    mistral_msg.append({"role":"user","content": chat_item})
                else:
                    mistral_msg.append({"role":"assistant","content": chat_item})
            mistral_args = {
                "model": args.model_name,
                "prompt": None,
                "messages": mistral_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            result = api(**mistral_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "anthropic":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = anthropic_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            system_msg = "You are a helpful AI assistant."
            anthropic_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    anthropic_msg.append({"role":"user","content": chat_item})
                else:
                    anthropic_msg.append({"role":"assistant","content": chat_item})
            anthropic_args = {
                "model": args.model_name.replace("anthropic/", ""),
                "prompt": None,
                "system_msg": system_msg,
                "messages": anthropic_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**anthropic_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    elif args.engine == "reka":
        todo_chats = chat_history[num_skipped:]
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = reka_chat_request(**kwargs)
            return result

        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"):
            # input_text = todo_inputs[cur_id]
            chat = todo_chats[cur_id]
            reka_msg = []
            for i, chat_item in enumerate(chat):
                if i % 2 == 0:
                    reka_msg.append({"role":"user","content": chat_item})
                else:
                    reka_msg.append({"role":"assistant","content": chat_item})
            reka_args = {
                "model": args.model_name.replace("reka/", ""),
                "messages": reka_msg,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }
            result = api(**reka_args)
            outputs.append(result)
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

