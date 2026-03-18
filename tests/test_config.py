from src.config_utils import get_shards_split
from src.config_parser import parse_args
from typing import List, Tuple
import pytest

def test_argparser():

    try:
        data_name = "mmlu-pro-short"
        args = parse_args(["--data_name",data_name])
    except Exception as e:
        pytest.fail(f"Config argument parser doesn't work: {e}")
    


test_data = [
(100,6, [(0, 17), (17, 34), (34, 51), (51, 68), (68, 84), (84, 100)]),
(110,1, [(0,110)]),
(2399,1, [(0,2399)]),
(2399,2, [(0,1200),(1200, 2399)])
]
@pytest.mark.parametrize("n_prompts,n_shards,expected_output", test_data)
def test_shards_split(n_prompts: int, n_shards: int, expected_output: List[Tuple[int,int]]):
    
    outs = get_shards_split(n_prompts, n_shards)
    assert outs == expected_output