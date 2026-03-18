import datasets
from datasets import load_dataset
import pytest
from huggingface_hub import login

from src.tasks import TASKS_COLLECTION

def test_load_ZebraLogicBench():
    try:
        dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    except Exception as e:
        pytest.fail(f"Loading dataset failed with exception: {e}")
    

test_data = [
("hendrycks-math"),
("mmlu-pro"),
("mmlu-pro-short"),
("arc-agi-2"),
]
@pytest.mark.parametrize("ds_name", test_data)
def test_hendrycks_math(ds_name):

    task = TASKS_COLLECTION[ds_name]
    dataset = task.load_dataset()

    assert len(dataset) == task.total_num_examples
    assert task.id_name in dataset[0].keys()