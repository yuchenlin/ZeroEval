
MMLU_PRO_PROMPT = """
The following are multiple choice questions (with answers) about {category} domain.
As an AI assistant with expertise across disciplines, think step by step using factual knowledge and logical reasoning. Avoid assumptions and refer to established concepts where possible.
After reasoning, output the final answer in JSON format.

{examples}

# Problem to Solve:

{question}

# Instruction

Now please solve the above problem step by step. Present your final answer in the following JSON format.

```json
{
    "answer": "$___$"
}
```
"""