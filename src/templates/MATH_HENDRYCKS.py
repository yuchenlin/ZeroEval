MATH_HENDRYCKS_PROMPT = r"""
Given a mathematics problem, determine the answer. Simplify your answer as much as possible.

Follow these guidelines:
- Show your reasoning clearly and logically.
- Double-check arithmetic and algebraic steps.
- The final answer must be placed in the JSON field `\"answer\"`.
- Always enclose the answer in \"$...$\" to ensure proper mathematical formatting (e.g., \"$\\sqrt{59}$\", \"$\\frac{1}{32}$\", \"$181$\").


Example 1:
Problem: What is $\left(\frac{7}{8}\right)^3 \cdot \left(\frac{7}{8}\right)^{-3}$?
Answer: $1$

Example 2:
Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
Answer: $15$

Example 3:
Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
Answer: $\sqrt{59}$

Example 4:
Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
Answer: $\frac{1}{32}$

Example 5:
Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
Answer: $181$

Example 6:
Problem: Calculate $6 \cdot 8\frac{1}{3}
Answer: $50$

Example 7:
Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
Answer: $2$

Example 8:
Problem: How many zeros are at the end of the product 25 $\times$ 240?
Answer: $3$


# Problem to Solve 

{problem}


# Instruction

Now please solve the above problem step by step. 
- Present your final answer in the following JSON format.
- The final answer must follow LaTeX formatting and must be enclosed in dollar signs: $...$ as shown in the template.

```json
{
    "answer": "$___$"
}
```

"""