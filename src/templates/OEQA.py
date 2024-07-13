#  Open-Ended Question Answering Template
OEQA = """
## Question: 

{question}


## Instruction 

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""
 