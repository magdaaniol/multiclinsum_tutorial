<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 Weasel Project: Prodigy DSPy Plugin Tutorial: Clinical Summarization

> **Note:** This tutorial requires Prodigy Company Plugins version 0.5.0 or later.

An end-to-end tutorial demonstrating the Prodigy-DSPy workflow for clinical
report summarization. This project follows the [blog post](https://explosion.ai/blog/human-aligned-llm-evaluation-dspy) on human-aligned LLM
evaluation and guides you through:
1. Annotating data with a baseline DSPy program
2. Evaluating and collecting human feedback on metrics
3. Synthesizing insights from feedback
4. Optimizing the program with human-in-the-loop guidance


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install all required dependencies |
| `download` | Download MultiClinSUM dataset from Zenodo and extract it |
| `preprocess` | Preprocess downloaded data into Prodigy format |
| `annotate` | Step 1: Annotate clinical notes with gold-standard summaries |
| `evaluate` | Step 2: Evaluate program and collect human feedback on metric quality |
| `process_feedback` | Step 3: Synthesize insights from human feedback under `human_feedback` field |
| `optimize` | Step 4: Optimize program with human-aligned feedback |
| `evaluate_test` | Step 5: Evaluate optimized program on held-out test set |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `setup` | `install` &rarr; `download` &rarr; `preprocess` |
| `full` | `install` &rarr; `download` &rarr; `preprocess` &rarr; `annotate` &rarr; `evaluate` &rarr; `process_feedback` &rarr; `optimize` &rarr; `evaluate_test` |
| `iterate` | `evaluate` &rarr; `process_feedback` &rarr; `optimize` &rarr; `evaluate_test` |

<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->