from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod
from datasets import Dataset
import datasets
import pandas as pd
from src.templates import (MATH_HENDRYCKS_PROMPT, MMLU_PRO_PROMPT)
import os
import random

# container for all existing tasks
TASKS_COLLECTION: Dict[str, TaskMeta] = {}

def register_class(cls: TaskMeta):
    # decorator for automatically registering classes
    TASKS_COLLECTION[cls.name] = cls()
    return cls


class TaskMeta(ABC):
    """A class that defines a task."""
    _id_name: str = "id" # column with uniqie_id of each dataset item
    name: str
    total_num_examples: int
    summary_file: str

    def __init__(self) -> "TaskMeta":
     pass

    def list_all_tasks(self):
        return ",".join([x.name for x in TASKS_COLLECTION])
            
    @abstractmethod
    def load_dataset(self) -> Dataset:
        # load hf dataset
        pass

    @property
    def id_name(self):
        return self._id_name
         
    def load_summary_file(self) -> pd.DataFrame:
        data = pd.read_json(self.summary_file)
        return data

    def get_output_dir(self) -> str:
         return os.path.join("result_dirs", self.name)

    def apply_template(self, item: dict) -> str:
        """ Apply Custom prompt template to Dataset item. """
        raise NotImplementedError(f"Method 'apply_template' not implemented for {self.name}")



@register_class
class GSMTask(TaskMeta):
     name = 'gsm'
     total_num_examples = 1319
     summary_file = 'result_dirs/gsm.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "gsm", split="test")
          return dataset

@register_class
class MMLUReduxTask(TaskMeta):
     name = 'mmlu-redux'
     total_num_examples = 2778
     summary_file = 'result_dirs/mmlu-redux.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
          return dataset
     

@register_class
class ZebraGridTask(TaskMeta):
     name = 'zebra-grid'
     total_num_examples = 1000
     summary_file = 'result_dirs/zebra-grid.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
          return dataset

@register_class
class AlpacaEval(TaskMeta):
     name = 'alpaca_eval'
     total_num_examples = -1
     summary_file = 'result_dirs/alpaca.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
          return dataset

@register_class
class NumerSenceV2Eval(TaskMeta):
     name = 'numersense-v2'
     total_num_examples = -1
     summary_file = 'result_dirs/numersense-v2.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("yuchenlin/zero-eval", "numersense-v2", split="test")
          return dataset

@register_class
class CruxTask(TaskMeta):
     name = 'crux'
     total_num_examples = 800
     summary_file = 'result_dirs/crux.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("flydust/zero-eval", "crux", split="test")
          return dataset

@register_class
class MathL5Task(TaskMeta):
     name = 'math-l5'
     total_num_examples = 721
     summary_file = 'result_dirs/math-l5.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
          return dataset
     
@register_class
class WildBench_V2Task(TaskMeta):
     _id_name = 'session_id'
     name = 'wildbench_v2-hard'
     total_num_examples = -1
     summary_file = 'result_dirs/wildbench_v2-hard.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("allenai/WildBench", "v2-hard", split="test")
          return dataset

@register_class
class GPlanetTask(TaskMeta):
     name = 'gplanet'
     total_num_examples = 1396
     summary_file = 'result_dirs/gplanet.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("WildEval/G-PlanET", split="test")
          return dataset


@register_class
class HendrycksMathTask(TaskMeta):
     _id_name: str = "unique_id"
     name = "hendrycks-math"
     total_num_examples = 500
     summary_file = 'result_dirs/gplanet.summary.json'

     def load_dataset(self):
          dataset = datasets.load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
          return dataset
          
     def apply_template(self, item: dict) -> str:
          prompt = MATH_HENDRYCKS_PROMPT
          prompt = prompt.replace("{problem}", item["problem"])
          return prompt
     


@register_class
class MMLUProTask(TaskMeta):
     _id_name: str = "question_id"
     name = "mmlu-pro"
     total_num_examples = 12032
     summary_file = 'result_dirs/mmlu-pro.summary.json'
     examples_by_category: Dict[set, Dataset] = None

     def load_dataset(self, seed=42):
          dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
          dataset = dataset.shuffle(seed=seed)
          return dataset
          

     @staticmethod
     def _format_example(question, options, cot_content=""):

          if cot_content.startswith("A: "):
               cot_content = cot_content[3:]

          example = "Question: {}\nOptions: \n".format(question)
          choice_map = "ABCDEFGHIJ"
          for i, opt in enumerate(options):
               example += "{}. {}\n".format(choice_map[i], opt)
          if cot_content == "":
               example += "\nAnswer: Let's think step by step... (think and return your answer with reasoning)"
          else:
               example += "\nAnswer: " + cot_content + "\n\n"
          return example
     

     def _load_cot_examples(self):
          if self.examples_by_category:
               return None
          
          examples = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation")               
          self.examples_by_category = {}
          for item in examples:
               cat = item['category']
               if cat not in self.examples_by_category:
                    self.examples_by_category[cat] = []
               self.examples_by_category[cat].append(item)
               

     def apply_template(self, item: dict) -> str:
          
          self._load_cot_examples()

          category = item['category']
               
          if category in self.examples_by_category:
               cnt = 0
               examples_prompt = ""
               for each in self.examples_by_category[category]:
                    cnt +=1 
                    examples_prompt += f"Example {cnt}:\n\n"
                    examples_prompt += self._format_example(each["question"], each["options"], each["cot_content"])


          prompt = MMLU_PRO_PROMPT.replace("{category}", category)
          prompt = prompt.replace("{examples}", examples_prompt)
          question = self._format_example(item["question"], item["options"])
          prompt = prompt.replace("{question}", question)

          return prompt
     

@register_class
class MMLUProShortTask(MMLUProTask):
    _id_name: str = "question_id"
    name = "mmlu-pro-short" # sampled from MMLU
    total_num_examples = 2399
    summary_file = 'result_dirs/mmlu-pro-short.summary.json'
    examples_by_category: Dict[set, Dataset] = None

    @staticmethod
    def _sample_dataset(dataset, sample_frac: float = 0.2, seed: int = 42):
        
        assert 0<sample_frac<=1

        # sample dataset by category
        cats: Dict[str, list] = {}
        for item in dataset:
            cat = item['category']
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(item)

        # sample each category with fixed seed
        random.seed(seed)
        for cat_name, questions in cats.items():
            random.shuffle(questions)
            n = max(10, int(len(questions) * sample_frac)) # at least 10 questions in category
            cats[cat_name] = questions[:n]

        del dataset

        dataset = []
        for cat in cats.values():
            dataset.extend(cat)
            
        del cats

        # shuffle 
        random.shuffle(dataset)

        return Dataset.from_list(dataset)


    def load_dataset(self, seed=42):
        dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        # sample 20% of the dataset
        dataset = self._sample_dataset(dataset, sample_frac = 0.2, seed = 42)

        return dataset
    




@register_class
class ArcAGI2Task(TaskMeta):
     _id_name: str = "arc_agi_2"
     name = "arc-agi-2"
     total_num_examples = 167 
     summary_file = 'result_dirs/arc-agi-2.summary.json'

     def load_dataset(self):
          # Note that github dataset has 1000 examples in train and 120 in test. 
          # But because some files contains several tests they were flattened.
          dataset = datasets.load_dataset("sirorezka/arc-agi-2", split="test")
          return dataset
          
     def apply_template(self, item: dict) -> str:

          1/0
          prompt = MATH_HENDRYCKS_PROMPT
          prompt = prompt.replace("{problem}", item["problem"])
          return prompt    