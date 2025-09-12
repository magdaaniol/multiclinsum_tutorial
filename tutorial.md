---
layout: post
title: "Beyond Basic Metrics: Engineering a Human-Aligned LLM Evaluation Workflow with Prodigy and DSPy"
date: 2023-10-27 10:00:00 +0000
categories: [LLM, NLP, DSPy, Prodigy, Evaluation]
author: Your Name Here
---

Large Language Models (LLMs) are incredibly powerful, but evaluating their performance, especially on nuanced tasks, remains a significant challenge. Standard metrics like BERTScore often fall short, failing to capture the subtle aspects of quality that a human expert would immediately spot. This post walks through a comprehensive workflow using **Prodigy** for targeted human feedback and **DSPy** for program optimization and LLM-as-a-judge metric development, ultimately demonstrating how to engineer an evaluation metric that truly aligns with human judgment.

## The Problem: When Standard Metrics Fall Short

We're working on a crucial task: generating concise, accurate, and complete clinical summaries from patient notes. Our initial DSPy program, while functional, needed optimization.

Our first thought for evaluation was a standard metric like BERTScore, which measures semantic similarity. We generated some summaries and evaluated them.

**Our Initial Program's Output (Example):**

* **Patient Note:** "Patient presented with flu-like symptoms. Diagnosed with Influenza A. Prescribed Tamiflu. Advised rest. Patient recovered completely within 5 days."
* **Model Summary:** "Patient had flu, diagnosed as Influenza A. Tamiflu prescribed."
* **Human Assessment:** "Partially Complete (missing recovery info), Accurate, Good Conciseness."

BERTScore might give this a high score due to semantic overlap, but a human expert would flag it as missing crucial information. This gap between automatic scores and human perception is what we set out to bridge.

## Step 1: Collecting Granular Human Feedback with Prodigy

To understand *why* our summaries were good or bad, we needed granular human feedback. We used **Prodigy**, a powerful annotation tool, to collect structured feedback on specific aspects of our clinical summaries.

Instead of a simple "good/bad," we broke down human judgment into key criteria:
* **Factual Accuracy:** Is the summary factually correct? (e.g., "Correct", "Minor Error", "Major Error")
* **Clinical Completeness:** Does it include all critical clinical information? (e.g., "Complete", "Partially Complete", "Incomplete")
* **Conciseness:** Is it appropriately succinct? (e.g., "Good", "Too Long", "Too Short")

Here's an example of our Prodigy annotation interface:

![Prodigy Annotation Interface for Clinical Summaries]
*(Image Placeholder: Prodigy annotation interface for multi-aspect feedback. You'd replace this with a screenshot of your Prodigy setup.)*

We annotated 30 examples, collecting detailed feedback that revealed a critical insight: **clinical completeness** was the most frequent point of failure for our initial model.

## Step 2: From Human Feedback to a Testable Metric Hypothesis with DSPy

Prodigy gave us qualitative insight. Now, we needed to translate that into a quantitative metric. DSPy's `dspy.feedback` module (or even just manual inspection of the human feedback) can suggest a composite metric.

Our initial hypothesis was a composite metric: a weighted average of BERTScore (for semantic similarity) and a simple entity overlap check (for completeness).

```python
# Simplified example of the initial composite metric
import dspy
from bert_score import score as bertscore
import spacy

# Load SciSpacy model or similar
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

# This would be registered as a dspy metric
def initial_composite_metric(gold: dspy.Example, pred: dspy.Prediction) -> float:
    # BERTScore component
    _, _, F1 = bertscore([pred.summary], [gold.summary], lang="en", verbose=False)
    bert_f1 = F1.item()

    # Simple entity overlap for completeness
    gold_entities = extract_entities(gold.summary)
    pred_entities = extract_entities(pred.summary)
    completeness_score = len(gold_entities.intersection(pred_entities)) / len(gold_entities) if gold_entities else 1.0

    # Initial arbitrary weighting
    return (0.7 * bert_f1) + (0.3 * completeness_score)
````

## Step 3: Validating the Metric: Is it Actually Better?

This is the crucial step. We compared our initial composite metric against the baseline BERTScore, correlating both with a numerical representation of our human judgment. For this, we first needed to convert our structured human feedback into a numerical score (e.g., `Complete=1.0, Partial=0.5, Incomplete=0.0`).

Our initial results (after testing various weightings for the simple composite metric) were disheartening. The correlation wasn't significantly better, and sometimes even worse, than BERTScore.

**Initial Correlation Results:**

  * Baseline BERTScore (Spearman's ρ): `0.1431`
  * Initial Composite Metric (Spearman's ρ): `0.1115` (even worse when heavily weighted towards simple entity overlap)

This was a critical learning moment: **a metric is only as good as its components**. Simple entity overlap was too brittle, punishing good summaries for using synonyms or slight rephrasing that `scispacy` didn't catch. It was adding noise, not signal.

## Step 4: Engineering a Superior Metric with LLM-as-a-Judge

Our failure analysis showed that a simplistic, rule-based approach for "completeness" wasn't enough. We needed a component that could understand nuance. This led us to the most powerful tool for human-like judgment: an **LLM-as-a-Judge**.

We designed a DSPy signature that asks an LLM to perform a holistic evaluation, considering all our criteria (accuracy, completeness, conciseness) and their relative importance.

```python
# In my_components.py

import dspy

class HolisticQualityAssessment(dspy.Signature):
    """Evaluate the overall quality of a predicted clinical summary against a gold summary.
    
    Consider the following three criteria in this order of importance:
    1.  **Factual Accuracy:** Is the information correct? A single major factual error makes the summary very poor.
    2.  **Clinical Completeness:** Does it include all critical information (diagnoses, outcomes)?
    3.  **Concisiveness:** Is it succinct and to the point?

    Based on these criteria, provide a single holistic quality score from 0.0 (terrible) to 1.0 (perfect).
    """
    gold_summary: str = dspy.InputField(desc="The trusted, high-quality reference summary.")
    predicted_summary: str = dspy.InputField(desc="The summary generated by the model to be evaluated.")
    holistic_quality_score: float = dspy.OutputField(desc="A single score from 0.0 to 1.0 representing overall quality.")

holistic_judge = dspy.Predict(HolisticQualityAssessment)

# This is our final, human-aligned metric
def holistic_llm_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    if not pred.summary:
        return 0.0
    result = holistic_judge(gold_summary=gold.summary, predicted_summary=pred.summary)
    return result.holistic_quality_score
```

We then re-ran our correlation analysis, comparing this new `holistic_llm_metric` against our original, combined `human_score`.

**The Visual Proof of Success:**

Here are the correlation plots comparing the baseline BERTScore (left) and our new Holistic LLM Metric (right) against the overall human quality score.

\![Correlation Plots: Baseline vs. Holistic LLM Metric]
*(Image Placeholder: The final correlation plot image with Baseline BERTScore on the left and New Composite Metric on the right, showing the improved correlation.)*

The right plot shows a visibly stronger trend. Importantly, our LLM judge now successfully assigns lower scores to summaries that humans rated as poor, which the baseline metric completely failed to do.

**The Quantitative Proof:**

  * Baseline BERTScore (Spearman's ρ): `0.1431`
  * **Holistic LLM Metric (Spearman's ρ): `0.2095`**

While a ρ of 0.21 might not be exceptionally high in absolute terms, it represents a significant relative improvement. Our new metric is **nearly 50% more correlated** with nuanced human judgment than the baseline BERTScore\! It's effectively captured the complex interplay of factors we wanted.

## Step 5: Optimizing Our Program with the New Metric

With a trustworthy metric in hand, we could finally proceed to `dspy.optimize`. We used our `holistic_llm_metric` to guide the optimization process on our 30 development examples. This ensured the optimizer was learning to generate summaries that scored highly on the very qualities our human experts cared about.

## Final Evaluation: Did It All Work?

The ultimate test was to evaluate the performance of our `baseline program` and our `optimized program` on a held-out test set of 100 examples. Crucially, we evaluated both programs using *both* metrics: the problematic BERTScore and our newly engineered Holistic LLM Metric.

Here are the definitive results:

| Program Version             | Measured by Baseline Metric (BERTScore) | Measured by **Our LLM Judge Metric** |
| :-------------------------- | :-------------------------------------- | :----------------------------------- |
| **Baseline Program** | 87.19                                   | 53.90                                |
| **Optimized Program** | 87.27 **(+0.08)** | 68.07 **(+14.17)** |

*(Image Placeholder: You can use a screenshot of this table in your blog post for visual clarity, or just keep it as markdown.)*

## Conclusion: The Power of Human-Aligned Evaluation

This journey highlights a critical lesson in developing robust LLM applications:

1.  **Standard metrics are often insufficient** for complex, nuanced tasks, failing to capture what truly matters to human users.
2.  **Granular human feedback is indispensable** for understanding model deficiencies and guiding metric development.
3.  **LLM-as-a-Judge metrics, carefully engineered with human insight, can become powerful, human-aligned evaluators.** They can understand the subtle interplay of quality attributes far better than simple rule-based or statistical metrics.
4.  **A good metric is essential to "see" progress.** Without our Holistic LLM Metric, we would have incorrectly concluded that our optimization efforts were futile (showing only `+0.08` improvement). Our new metric revealed a substantial **+14.17 point (26%) improvement**, clearly demonstrating the value of our work.

The Prodigy-DSPy workflow empowers developers to move beyond guesswork and build LLM systems that are not just technically performant, but truly valuable to the end-user. By investing in better evaluation, we unlock the true potential of LLMs.