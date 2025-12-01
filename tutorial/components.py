"""
Prodigy DSPy Tutorial: Clinical Summarization Components

This module contains all task-specific components for the clinical summarization
workflow, including DSPy programs, metrics, converters, evaluators, and UI configs.
"""

import json
import dspy
import spacy
from typing import Callable, Dict, Any, Iterator, Optional, List
from prodigy.util import registry, set_hashes
from prodigy_company_plugins.prodigy_dspy.evaluators import BaseTaskEvaluator
from prodigy.types import StreamType
from wasabi import msg
import copy


# ============================================================================
# DSPy Program - Baseline Summarization
# ============================================================================


class ClinicalSummarySignature(dspy.Signature):
    """Summarize a clinical case report with key clinical information."""

    text: str = dspy.InputField(
        desc="The clinical case report to summarize."
    )
    summary: str = dspy.OutputField(
        desc="A concise summary of the clinical case."
    )


class ClinicalSummarizationProgram(dspy.Module):
    """
    Baseline DSPy program for clinical summarization.

    Uses chain-of-thought prompting to encourage the model to reason through
    the key clinical elements before generating a summary.
    """

    def __init__(self):
        super().__init__()
        self.generate_summary = dspy.ChainOfThought(ClinicalSummarySignature)

    def forward(self, text: str):
        return self.generate_summary(text=text)


@registry.dspy_programs.register("clinical_summarization.v1")
def make_clinical_summarization_program(load_path: Optional[str] = None):
    """
    Baseline DSPy program for clinical summarization using chain-of-thought prompting
    to reason through key clinical elements before generating a summary.
    """
    program = ClinicalSummarizationProgram()
    if load_path:
        msg.info(f"Loading program weights from {load_path}")
        program.load(load_path)
    return program


# ============================================================================
# Data Converters - Transform between Prodigy and DSPy formats
# ============================================================================


@registry.dspy_converters.register("dspy_to_prodigy.clinical_summary.v1")
def make_dspy_to_prodigy_converter():
    """
    Convert DSPy prediction to Prodigy task format for annotation UI.

    This converter prepares tasks for the annotate step where users create
    gold-standard summaries. The model prediction is used to pre-fill the
    text input, allowing users to edit it or write from scratch.
    """

    def dspy_to_prodigy(task: Dict, prediction: dspy.Prediction) -> Dict:
        """
        Add the model's predicted summary to the task for user review.
        Args:
            task: Prodigy task dict with clinical note
            prediction: DSPy model prediction

        Returns:
            Task dict with pred_summary and gold_summary fields
        """
        task_copy = copy.deepcopy(task)

        # Store the raw model prediction for reference
        task_copy["pred_summary"] = getattr(prediction, "summary", "")

        # Pre-fill the gold_summary field - users will edit this to create
        # the gold standard. Starting from the model prediction speeds up
        # annotation by allowing users to edit rather than write from scratch.
        task_copy["gold_summary"] = getattr(prediction, "summary", "")

        # Add reasoning if available (useful for understanding model's logic)
        if hasattr(prediction, "reasoning"):
            task_copy["reasoning"] = prediction.reasoning

        return task_copy

    return dspy_to_prodigy


@registry.dspy_converters.register("prodigy_to_dspy.clinical_summary.v1")
def make_prodigy_to_dspy_converter():
    """
    Convert Prodigy annotation to DSPy example format.
    """

    def prodigy_to_dspy(eg: Dict) -> dspy.Example:
        """
        Convert a Prodigy example to DSPy format.

        Args:
            eg: Prodigy example from database

        Returns:
            DSPy Example with text input and summary output, optionally with feedback
        """
        # Create DSPy example with clinical note as input
        example = dspy.Example(
            text=eg.get("text", ""),
            summary=eg.get("gold_summary", "")
        )

        # Attach human feedback if present
        if "human_feedback" in eg and eg["human_feedback"]:
            feedback_str = json.dumps(eg["human_feedback"], indent=2)
            example = example.copy(feedback=feedback_str)
        return example.with_inputs("text")

    return prodigy_to_dspy


# ============================================================================
# Metric Functions
# ============================================================================


@registry.dspy_metrics.register("baseline_bertscore.v1")
def make_baseline_bertscore_metric():
    """
    Baseline metric using BERTScore.
    """
    try:
        from bert_score import score
    except ImportError:
        msg.fail(
                "BERTScore not installed. Install with: pip install bert-score", exits=1
            )
    def baseline_bertscore(
        gold: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> float:
        """
        Calculate BERTScore between predicted and gold summary.

        Args:
            gold: Gold-standard example with reference summary
            pred: Model prediction
            trace: Execution trace (optional)

        Returns:
            BERTScore F1 value (0.0-1.0)
        """
        gold_summary = getattr(gold, "summary", "")
        pred_summary = getattr(pred, "summary", "")

        if not pred_summary or not gold_summary:
            return 0.0

        try:
            _, _, F1 = score(
                [pred_summary], [gold_summary], lang="en", verbose=False
            )
            return float(F1.item())
        except Exception as e:
            msg.warn(f"BERTScore calculation failed: {e}")
            return 0.0

    return baseline_bertscore


def extract_entities(text: str, nlp) -> set:
    """Extract medical entities from text using scispacy."""
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)


@registry.dspy_metrics.register("composite_ner_metric.v1")
def make_composite_ner_metric():
    """
    Composite metric combining BERTScore and NER-based entity overlap.

    This metric addresses clinical completeness by checking if key clinical
    entities from the gold summary appear in the predicted summary.
    """
    nlp = spacy.load("en_core_sci_lg")
    def composite_ner_metric(
        gold: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> float:
        """
        Calculate composite score based on semantic similarity (BERTScore) and entity overlap (clinical completeness).

        Args:
            gold: Gold-standard example
            pred: Model prediction
            trace: Execution trace (optional)

        Returns:
            Composite score (0.0-1.0) = 0.7 * BERTScore + 0.3 * Entity overlap
        """
        try:
            from bert_score import score
        except ImportError:
            msg.fail(
                "BERTScore not installed. Install with: pip install bert-score", exits=1
            )

        gold_summary = getattr(gold, "summary", "")
        pred_summary = getattr(pred, "summary", "")

        if not pred_summary or not gold_summary:
            msg.warn("Empty task, returning score of 0.0")
            return 0.0

        # Component 1: BERTScore for semantic similarity
        try:
            _, _, F1 = score(
                [pred_summary], [gold_summary], lang="en", verbose=False
            )
            bert_score = float(F1.item())
        except Exception:
            bert_score = 0.0

        # Component 2: Entity overlap for clinical completeness
        try:
            gold_entities = extract_entities(gold_summary, nlp)
            pred_entities = extract_entities(pred_summary, nlp)

            if not gold_entities:
                completeness_score = 1.0
            else:
                overlap = len(gold_entities.intersection(pred_entities))
                completeness_score = overlap / len(gold_entities)
        except Exception:
            completeness_score = 0.0

        # Combine: 70% semantic, 30% completeness
        composite_score = (0.7 * bert_score) + (0.3 * completeness_score)
        return composite_score

    return composite_ner_metric


class HolisticQualityAssessment(dspy.Signature):
    """
    Evaluate the overall quality of a clinical summary.

    Consider three criteria in order of importance:
    1. Factual Accuracy: Is the information correct? Major errors make it very poor.
    2. Clinical Completeness: Does it include critical diagnoses and interventions?
    3. Conciseness: Is it succinct and focused?
    Evaluate the predicted summary along these three dimensions, scoring each from 0 to 1:

    1. Factual accuracy (0-1)
    2. Clinical completeness (0-1)
    3. Conciseness (0-1)

    Then compute holistic_quality_score = 0.5*factual + 0.3*completeness + 0.2*conciseness
    """

    gold_summary: str = dspy.InputField(
        desc="The high-quality reference summary."
    )
    predicted_summary: str = dspy.InputField(
        desc="The summary generated by the model to be evaluated."
    )
    factual_score: float = dspy.OutputField(
        desc="Factual accuracy score (0.0-1.0)."
    )
    completeness_score: float = dspy.OutputField(
        desc="Clinical completeness score (0.0-1.0)."
    )
    conciseness_score: float = dspy.OutputField(
        desc="Conciseness score (0.0-1.0)."
    )
    holistic_quality_score: float = dspy.OutputField(
        desc="Final holistic score computed as: 0.5*factual + 0.3*completeness + 0.2*conciseness"
    )


@registry.dspy_metrics.register("holistic_llm_metric.v1")
def make_holistic_llm_metric():
    """
    LLM-as-a-judge metric for human-aligned evaluation.

    Uses a powerful LLM to evaluate summaries holistically, considering factual
    accuracy, clinical completeness, and conciseness with appropriate weights.
    """

    def holistic_llm_metric(
        gold: dspy.Example, pred: dspy.Prediction, trace=None
    ) -> float:
        """
        Use an LLM to evaluate summary quality.

        Args:
            gold: Gold-standard example
            pred: Model prediction
            trace: Execution trace (optional)

        Returns:
            Holistic quality score (0.0-1.0)
        """
        holistic_judge = dspy.Predict(HolisticQualityAssessment)

        gold_text = getattr(gold, "summary", "")
        pred_text = getattr(pred, "summary", "")

        if not gold_text or not pred_text:
            return 0.0

        try:
            result = holistic_judge(gold_summary=gold_text, predicted_summary=pred_text)
        except Exception as e:
            msg.warn(f"LLM judge invocation failed: {e}, {gold_text}, {pred_text}")
            return 0.0
        
        score = getattr(result, "holistic_quality_score", None)
        try:
            return float(score)
        except (TypeError, ValueError):
            msg.warn(f"Invalid score from LLM judge: {score}")
            return 0.0

    return holistic_llm_metric


# ============================================================================
# Task Evaluator - stream for evaluation feedback
# ============================================================================


class ClinicalSummarizationEvaluator(BaseTaskEvaluator):
    """
    Evaluator for the clinical summarization task.

    Converts evaluation results into Prodigy task format for the metric debugging UI,
    and, optionally, handles post-processing of human feedback from annotations.
    """

    def get_stream(
        self, results: dspy.evaluate.evaluate.EvaluationResult
    ) -> StreamType:
        """
        Convert evaluation results to a stream of Prodigy tasks.

        Args:
            results: Evaluation result from dspy.Evaluate

        Yields:
            Prodigy task dicts for the evaluation UI
        """
        for example, prediction, score in results.results:
            task = {
                "page_titles": ["Factual Accuracy", "Clinical Completeness", "Conciseness"],
                "pages":[
                    {"text": example.text,
                     "gold_summary": example.summary,
                     "pred_summary": prediction.summary,
                     "metric_score": round(score, 3),
                     "options": [
                         {"id": "correct", "text": "Correct"},
                         {"id": "minor_error", "text": "Minor Error"},
                         {"id": "major_error", "text": "Major Error"},
                     ],
                     "view_id": "blocks",
                     "config": {
                         "blocks": [
                                {"view_id": "html",
                                 "html_template": "<h3>Clinical Note</h3><p>{{text}}</p>"},
                                {"view_id": "html",
                                 "html_template": "<h3>Gold-Standard Summary</h3><p><strong>{{gold_summary}}</strong></p>"},
                                {"view_id": "html",
                                 "html_template": "<h3>Model-Generated Summary</h3><p>{{pred_summary}}</p>"},
                                {"view_id": "html",
                                 "html_template": "<h3>Metric Score: {{metric_score}}</h3>"},
                                {"view_id": "choice",
                                 "text": None,
                                 "html": None,
                                },
                                {"view_id": "text_input", "field_id": "comment", "field_label": "Comments (optional)",}
                            ]
                     }
                    },
                    {
                        "text": example.text,
                        "gold_summary": example.summary,
                        "pred_summary": prediction.summary,
                        "metric_score": round(score, 3),
                        "options": [
                            {"id": "complete", "text": "Complete"},
                            {"id": "partial", "text": "Partially Complete"},
                            {"id": "incomplete", "text": "Incomplete"},
                        ],
                        "view_id": "blocks",
                        "config": {
                            "blocks": [
                                   {"view_id": "html",
                                    "html_template": "<h3>Clinical Note</h3><p>{{text}}</p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Gold-Standard Summary</h3><p><strong>{{gold_summary}}</strong></p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Model-Generated Summary</h3><p>{{pred_summary}}</p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Metric Score: {{metric_score}}</h3>"},
                                   {"view_id": "choice",
                                    "text": None,
                                    "html": None,
                                   },
                                   {"view_id": "text_input", "field_id": "comment", "field_label": "Comments (optional)",}
                               ]
                        }
                    },
                    {
                        "text": example.text,
                        "gold_summary": example.summary,
                        "pred_summary": prediction.summary,
                        "metric_score": round(score, 3),
                        "options": [
                            {"id": "good", "text": "Good"},
                            {"id": "too_long", "text": "Too Long"},
                            {"id": "too_short", "text": "Too Short"},
                        ],
                        "view_id": "blocks",
                        "config": {
                            "blocks": [
                                   {"view_id": "html",
                                    "html_template": "<h3>Clinical Note</h3><p>{{text}}</p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Gold-Standard Summary</h3><p><strong>{{gold_summary}}</strong></p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Model-Generated Summary</h3><p>{{pred_summary}}</p>"},
                                   {"view_id": "html",
                                    "html_template": "<h3>Metric Score: {{metric_score}}</h3>"},
                                   {"view_id": "choice",
                                    "text": None,
                                    "html": None,
                                   },
                                   {"view_id": "text_input", "field_id": "comment", "field_label": "Comments (optional)",}
                               ]    
                    }
                    }
                ],
            }
            yield set_hashes(task)

    def get_before_db_callback(self) -> Optional[Callable]:
        """
        Return a callback to post-process annotations before saving to DB.

        This callback structures the raw UI data into the format expected by
        downstream recipes (dspy.feedback and dspy.optimize):
        - human_feedback: Structured feedback from the UI choices and comments
        - metric_score: The metric's score for this example

        These fields are required for the feedback synthesis and optimization workflows.
        """
        def before_db_callback(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            for eg in examples:
                # Extract metric_score from the first page (all pages have the same score)
                pages = eg.get("pages", [])
                if pages:
                    eg["metric_score"] = pages[0].get("metric_score", 0.0)
                    # Also preserve text and summaries at top level for easier access
                    eg["text"] = pages[0].get("text", "")
                    eg["gold_summary"] = pages[0].get("gold_summary", "")
                    eg["pred_summary"] = pages[0].get("pred_summary", "")

                # Structure human feedback from the pages UI
                eg["human_feedback"] = _create_feedback_summary(eg)

            return examples

        return before_db_callback


@registry.dspy_evaluators.register("clinical_summarization_evaluator.v1")
def make_clinical_summarization_evaluator(
    num_threads: Optional[int] = None
) -> ClinicalSummarizationEvaluator:
    """Factory function to create the clinical summarization evaluator.

    Note: program and metric are injected by the recipe via set_program/set_metric.
    This factory only takes configuration parameters.
    """
    return ClinicalSummarizationEvaluator(num_threads=num_threads)


# ============================================================================
# UI Configuration for Annotation and Evaluation
# ============================================================================


@registry.dspy_ui.register("clinical_annotation_ui.v1")
def make_clinical_annotation_ui() -> Dict[str, Any]:
    """
    UI for annotating clinical summaries.

    This interface supports creating gold-standard summaries by editing
    model predictions. 
    Displays:
    - The original clinical note (read-only HTML)
    - Text input prepopulated with model prediction for editing
    - Model's reasoning chain-of-thought (collapsible, if available)
    """
    custom_css = '''
    .prodigy-container { 
        max-width: 1500px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: auto 1fr auto;
        gap: 20px;
        height: 85vh;
        grid-template-areas: "doc summary" "doc summary" "reasoning reasoning";
        }
    div.prodigy-content:nth-child(2) { grid-area: doc; overflow-y: auto; }
    div.prodigy-content:nth-child(3) { grid-area: summary; }
    div.prodigy-content:nth-child(4) { grid-area: reasoning; overflow-y: auto; }
    .prodigy-container>.prodigy-content { white-space: normal; border-top: 1px solid #ddd; }
    .cleaned { text-align: left; font-size: 14px; }
    .cleaned pre { background-color: #eee; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 15px 20px; border-radius: 15px; white-space: pre-wrap; }
    summary { font-weight: bold; cursor: pointer; font-size: 1.2em; }
    details { margin-bottom: 1rem; }
    '''
    return {
        "view_id": "blocks",
        "config":{
            "global_css": custom_css,
            "blocks": [
                {
                    "view_id": "html",
                    "html_template": "<h3>Clinical Note</h3><p>{{text}}</p>",
                },
                {
                    "view_id": "text_input",
                    "field_id": "gold_summary", # Prepopulate with model summary for editing
                    "field_label": "Gold-Standard Summary (Edit or write your own)",
                    "field_rows": 30,
                },
                {
                    "view_id": "html",
                    "html_template": "<div class='cleaned'><details><summary>Reasoning</summary><pre>{{reasoning}}</pre></details></div",
                },
            ]
        }
    }


@registry.dspy_ui.register("clinical_evaluation_ui.v1")
def make_clinical_evaluation_ui() -> Dict[str, Any]:
    """
    UI for evaluating summaries and collecting human feedback.

    Displays:
    - Clinical note
    - Gold-standard summary
    - Model prediction
    - Metric score
    - Feedback collection with structured choices and free text
    """
    custom_css = '''
    .prodigy-container { max-width: 1500px; }
    .prodigy-page-content{
        max-width: 1500px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: minmax(300px, auto) auto auto auto auto;
        gap: 20px;
        height: auto;
        min-height: 85vh;
        grid-template-areas: "document document" "gold-summary model-summary" "score score" "options options" "notes notes";
        padding: 20px;
        }
    .prodigy-page-content > .prodigy-content:nth-child(1) { grid-area: document; overflow-y: auto; max-height: 25vh; }
    .prodigy-page-content > .prodigy-content:nth-child(2) { grid-area: gold-summary; }
    .prodigy-page-content > .prodigy-content:nth-child(3) { grid-area: model-summary; }
    .prodigy-page-content > .prodigy-content:nth-child(4) { grid-area: score; }
    .prodigy-options { grid-area: options; }
    .prodigy-page-content > .prodigy-content:last-child { grid-area: notes !important; }
    .prodigy-page-content > * { position: static !important; float: none !important; clear: both !important; margin: 0 !important; }
    .prodigy-container>.prodigy-content { white-space: normal; border-top: 1px solid #ddd; }
    '''
    return {
        "view_id": "pages",
        "config": {
            "global_css": custom_css
        }
    }

def _create_feedback_summary(eg: Dict) -> Dict:
    """
    Parse the pages UI structure and return a clean feedback dictionary.

    Extracts structured feedback choices and comments from each page of the
    evaluation UI into a flat dictionary format suitable for LLM synthesis.

    Args:
        eg: Prodigy example with pages structure from evaluate recipe

    Returns:
        Dictionary with feedback organized by aspect (e.g., factual_accuracy,
        clinical_completeness, conciseness)
    """
    feedback = {}
    page_titles = eg.get("page_titles", [])
    pages = eg.get("pages", [])

    for i, page_data in enumerate(pages):
        # Convert page title to dict key (e.g., "Factual Accuracy" -> "factual_accuracy")
        title = page_titles[i] if i < len(page_titles) else f"aspect_{i+1}"
        key = title.lower().replace(" ", "_")

        # Extract choice selection (e.g., "correct", "minor_error", "major_error")
        answer = page_data.get("accept", [])
        if answer:
            feedback[key] = answer[0]

        # Extract optional comment
        comment = page_data.get("comment")
        if comment:
            feedback[f"{key}_comment"] = comment

    return feedback




