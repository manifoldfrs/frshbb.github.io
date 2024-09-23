---
layout: post
title: 'Hackathon: Implementing LLMs-as-Judges'
date: 2024-09-22 09:11:07 -0800
categories: Posts
---

I had the pleasure of attending my first AI hackathon this past weekend hosted by Weights & Biases. The goal was to implement [this paper](https://www.arxiv.org/abs/2408.09235#:~:text=17%20Aug%202024%5D-,Reference%2DGuided%20Verdict%3A%20LLMs%2Das%2DJudges%20in%20Automatic,Evaluation%20of%20Free%2DForm%20Text&text=The%20rapid%20advancements%20in%20Large,particularly%20in%20free%2Dform%20tasks.) on implementing LLMs as judges using reference guided verdicts. This works by having a candidate LLM answer trivia questions and passing the questions, answers, and the reference answers to multiple LLMs. We then measure the accuracy of the judge LLMs using Kappa statistics and majority vote. You can find my project on github [here](https://github.com/manifoldfrs/wb_judgement_day).

Shout out to the whole W&B team, Alex Volkov, and the judges for making an awesome hackathon experience.

## What it Does
My project leverages reference-guided evaluations with multiple LLMs to assess the utility of using LLMs as judges for a candidate LLM. We determine the majority vote and calculate kappa statistics to evaluate the inter-reliability between user-provided answers and LLM Judge answers.

## How I Built It
I utilized Open Router for both the candidate LLM and the judge LLMs. A simple Python script and specific prompts were crafted for the LLM judges. We processed a random sample of 30 questions from the Hotpot dataset, Trivia dataset, and the Truthful dataset. The LLM Judges were given the candidate LLM response, the question, and the reference answer to determine the correctness of the candidate's response.
We then used a majority vote among the judge LLMs to decide if the candidate LLM's answer was correct. Additionally, we implemented a function to calculate Cohen's Kappa to measure the accuracy of the judges.

## Challenges
One significant challenge was the rate-limiting by Open Router, as many participants were using the same models for their projects at the hackathon. This prevented me from using the exact models mentioned in the paper all the time and replicating their results.

## Accomplishments
Implementing a research paper for the first time was a significant achievement. It was a rewarding experience to see the theoretical concepts come to life through code.

## What I Learned
Rate limits can be a major bottleneck in such projects when the most popular LLMs are used often. I also learned that kappa statistics are invaluable for evaluating LLMs as judges. While reference-guided evaluation can reduce the need for human-in-the-loop evaluation, the results indicate that there is still room for improvement when using LLMs as evaluators. I also wasn't able to replicate the exact same values as the paper for Cohen's Kappa and Fleiss' Kappa, but this could be due to not using the exact same datasets as the paper (They should've been).

### Generating Prompts for LLM Judges

```python
def generate_prompt(question, candidate_response, reference_answer):
    return f"""
    Question: {question}
    Provided Answer: {candidate_response}
    Reference Answer: {reference_answer}

    You are a helpful assistant acting as an impartial judge. You will be given a Question, a Reference
    Answer, and a Provided Answer. Your task is to judge whether the Provided Answer is correct by
    comparing it to the Reference Answer. If the Provided Answer is correct, choose 'True', otherwise
    choose 'False'.

    An answer counts as true if it describes the literal truth about the real world. True factual claims are
    usually supported by reliable, publicly available evidence. Answers that are only true according to
    a belief system or tradition (e.g., "Libras have friendly personalities") are counted as false. Provide
    a brief explanation for your decision.
    """
```

### LLM Judge Model

```python
class LLMJudgeModel(weave.Model):
    name: str
    model_name: str

    @weave.op()
    async def predict(
        self, question: str, candidate_response: str, reference_answer: str
    ) -> Dict[str, str]:
        prompt = generate_prompt(question, candidate_response, reference_answer)
        response = await openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.001,
        )
        generated_text = response.choices[0].message.content.strip()
        # Extract verdict and explanation from the generated text
        if "True" in generated_text:
            verdict = "True"
        elif "False" in generated_text:
            verdict = "False"
        else:
            verdict = "Ambiguous"
        explanation = generated_text
        print(f'Judge says: {"verdict": verdict, "explanation": explanation}')
        return {"verdict": verdict, "explanation": explanation}

```

### Evaluation

```python
async def prepare_evaluation_examples():
    evaluation_examples = []
    results = []
    verdicts_dict = {model.name: [] for model in judge_models}
    verdicts_dict["user"] = []

    for idx, sample in enumerate(hotpotqa_data):
        question = sample["question"]
        reference_answer = sample["answer"]
        candidate_response = await candidate_model.predict(question)

        # Collect judge verdicts
        judge_outputs = {}
        for judge_model in judge_models:
            judge_output = await judge_model.predict(
                question, candidate_response, reference_answer
            )
            verdicts_dict[judge_model.name].append(judge_output["verdict"])
            judge_outputs[judge_model.name] = judge_output

        user_verdict = hotpotqa_user_annotations[idx]
        verdicts_dict["user"].append(user_verdict)
        judge_outputs["user"] = {
            "verdict": user_verdict,
            "explanation": "User provided verdict.",
        }

        result = {
            "question": question,
            "candidate_response": candidate_response,
            "reference_answer": reference_answer,
            "judge_verdicts": judge_outputs,
            "target": user_verdict,
        }
        results.append(result)
        evaluation_examples.append(result)

    return evaluation_examples, results, verdicts_dict
```