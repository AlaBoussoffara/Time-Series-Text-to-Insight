import os
from pydantic import BaseModel, Field
from typing import Optional
from utils.general_helpers import llm_from

class JudgeResult(BaseModel):
    score: int = Field(description="Score from 0 to 10")
    reasoning: str = Field(description="Explanation for the score")

class BenchmarkJudge:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        if provider or model:
            resolved_provider = provider or os.getenv("USE_PROVIDER", "aws")
            resolved_model = model or os.getenv(
                "USE_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"
            )
            self.llm = llm_from(resolved_provider, resolved_model).with_structured_output(JudgeResult)
        else:
            self.llm = llm_from().with_structured_output(JudgeResult)

    def judge_sql(self, question: str, ground_truth_sql: str, generated_sql: str) -> JudgeResult:
        """Evaluates the generated SQL against the ground truth."""
        
        prompt = f"""
        You are an expert SQL judge. Evaluate the generated SQL query against the ground truth query for a given question.
        
        Question: {question}
        
        Ground Truth SQL:
        ```sql
        {ground_truth_sql}
        ```
        
        Generated SQL:
        ```sql
        {generated_sql}
        ```
        
        Compare them based on:
        1. Semantic equivalence: Does the generated SQL query return the same result set as the ground truth?
        2. Correctness: Is the generated SQL valid syntax and logic?
        3. Efficiency/Style: (Minor factor) Is it reasonable SQL?
        
        Rate on a scale of 0 to 10:
        - 10: Perfect match or semantically identical equivalent.
        - 8-9: Correct logic but slightly different formatting or minor inefficiencies.
        - 5-7: Returns correct data but with extra columns, wrong order, or correct logic but invalid syntax (if minor).
        - 1-4: Wrong logic, wrong table/columns, or syntax error.
        - 0: Completely irrelevant or no SQL generated (N/A).
        
        If Generated SQL is "N/A" or empty, score is 0.
        """
        if generated_sql in ["N/A", "", None, "ERROR"]:
            return JudgeResult(score=0, reasoning="No SQL was generated.")

        return self.llm.invoke(prompt)

    def judge_answer(self, question: str, ground_truth_sql: str, supervisor_answer: str) -> JudgeResult:
        """Evaluates the supervisor's final answer."""
        
        prompt = f"""
        You are an expert judge evaluating an AI agent's answer to a database question.
        
        Question: {question}
        
        The question implies a query that would produce certain results.
        Ground Truth SQL (defines the correct data source/logic):
        ```sql
        {ground_truth_sql}
        ```
        
        Supervisor Agent's Final Answer:
        "{supervisor_answer}"
        
        Evaluate the Answer:
        1. Accuracy: Does the text answer correctly interpret the data that would be returned by the Ground Truth SQL?
        2. Completeness: Does it answer the user's specific question?
        3. Hallucination: Does it invent facts not supported by the SQL context (assuming the SQL worked)?
        
        Note: The Supervisor might have run a DIFFERENT but valid SQL. Focus on whether the ANSWER answers the question correctly. 
        If the Supervisor says "I couldn't find the data" but the data exists (per Ground Truth), that is a low score.
        If the data doesn't exist (e.g. table missing) and Supervisor says so, that is a high score.
        
        Rate on a scale of 0 to 10:
        - 10: Perfect, accurate, and complete answer.
        - 8-9: Correct answer but minor details missing or verbosity.
        - 5-7: Partially correct, or correct but vague.
        - 1-4: Incorrect facts, or mostly failure to answer.
        - 0: Complete hallucinations or failure.
        """
        
        if not supervisor_answer:
            return JudgeResult(score=0, reasoning="No answer provided.")

        return self.llm.invoke(prompt)
