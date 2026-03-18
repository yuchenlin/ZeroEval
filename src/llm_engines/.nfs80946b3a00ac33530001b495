from argparse import Namespace
import asyncio
from asyncio import Queue, QueueEmpty
from typing import List, Any, Callable, TYPE_CHECKING
from tqdm import tqdm
import random
from src.llm_engines.models import EngineProto
from src.config_parser import RunConfig

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.core_client import DPLBAsyncMPClient
except:
    pass

from vllm import AsyncLLMEngine

def create_vllm_async_engine(args: RunConfig):
    max_model_len = None if args.max_model_len == -1 else args.max_model_len
    engine_args = AsyncEngineArgs(
        model=args.model_name,
        tokenizer=args.tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        dtype=args.dtype,
        # gpu_memory_utilization=args.gpu_memory_utilization, # gives errors
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    # llm = None
    llm = AsyncLLMEngine.from_engine_args(engine_args)    
    return llm


async def _shutdown_engine(engine: AsyncLLMEngine | None):
    if engine is None:
        return
    
    try:
        engine.shutdown()
        print("vLLM AsyncEngine shut down gracefully.")
    except Exception as e:
        print(f"Error when closing engind: {e}")


def shutdown_vllm_async_engine(llm):
    asyncio.run(_shutdown_engine(llm))


# Define worker function
async def worker(llm: AsyncLLMEngine, 
                 sampling_params: SamplingParams, 
                 prompts: List[str], 
                 outputs: List[List[str]], 
                 dp_queue_size: List[int],
                 queue: Queue, 
                 pbar: tqdm, 
                 worker_id: int = 0,
                 saver: Callable | None = None):
    """
    Args:
        saver (Callable | None, optional): function that dumps generated outputs to selected file.

    Raises:
        e: _description_
    """


    save_interval = 10 # save every 10 generations

    while not queue.empty():

        try:
            prompt_idx = queue.get_nowait()
            prompt = prompts[prompt_idx]
        except QueueEmpty as e:
            break
        
        # simple load balancing
        dp_idx = 0
        for i in range(len(dp_queue_size)):
            if dp_queue_size[i] < dp_queue_size[dp_idx]:
                dp_idx = i

        # print(f"worker_id: {worker_id}", dp_idx, dp_queue_size)
        dp_queue_size[dp_idx] += 1

        try:
            request_id = f"request-{prompt_idx}"
            # Get a prompt idx from the queue
            generator = llm.generate(prompt, 
                                     sampling_params, 
                                     request_id=request_id,
                                     data_parallel_rank = dp_idx)

            final_result = None
            async for result in generator:
                final_result = result
            
            outs = [x.text for x in final_result.outputs]
            outputs[prompt_idx].extend(outs)

            pbar.update(1)
            if (saver is not None) and (pbar.n % save_interval == 0):
                saver(outputs)
                
        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            raise e
        finally:
            dp_queue_size[dp_idx] -= 1
            queue.task_done()


async def _run_async_inference(llm, 
                               args: RunConfig, 
                               sampling_params, 
                               prompts, 
                               num_workers = 10,
                               saver: Callable | None = None):
    n = len(prompts)
    num_workers = min(n, num_workers)

    dp_queue_size = [0 for _ in range(args.data_parallel_size)] 

    workers = []
    outputs = [[] for _ in range(n)]  # prepopulate outputs
    descr = f"Generating {args.model_name} from {args.start_index} to {args.end_index} on {args.data_name}"
    with tqdm(total=n, desc=descr) as pbar:

        # Create an async queue and populate it with prompt indexes
        queue = asyncio.Queue(maxsize=n)
        for i in range(len(prompts)):
            queue.put_nowait(i)

        # Create and start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(worker(llm, sampling_params, prompts, outputs, 
                                              dp_queue_size = dp_queue_size, 
                                              queue = queue, 
                                              pbar=pbar, 
                                              worker_id=i, 
                                              saver=saver))
            workers.append(task)
        
        await asyncio.gather(*workers, return_exceptions=True)

    for task in workers:
        if task.exception():
            raise task.exception()
        
    return outputs


def run_vllm_async_inference(llm, 
                             args: RunConfig, 
                             sampling_params: SamplingParams, 
                             prompts: List[str],
                             saver: Callable | None = None):
    """
    Run inference using the async vLLM engine.
    Prompts are processed by async workers that send them to the engine.

    Args:
        prompts - are rendered prompts, e.g. chat template was applied to them.

    """
    if len(prompts) == 0:
        return []
    
    num_workers = 10 * args.data_parallel_size
    print(f"Num async workers {num_workers}")
    outputs = asyncio.run(_run_async_inference(llm, args, sampling_params, prompts, num_workers=num_workers, saver=saver))
    return outputs

