# MultiClinSUM Tutorial

This repository contains companion code for the blog post [Human-Aligned LLM Evaluation with DSPy](https://explosion.ai/blog/human-aligned-llm-evaluation-dspy).

The tutorial uses data from the [MultiClinSUM](https://temu.bsc.es/multiclinsum/) shared task, a multilingual clinical report summarization challenge organized by the Barcelona Supercomputing Center's [NLP for Biomedical Information Analysis group](https://www.bsc.es/discover-bsc/organisation/research-departments/nlp-biomedical-information-analysis).

## Contents

The `tutorial/` directory contains a complete end-to-end tutorial demonstrating the Prodigy-DSPy workflow for clinical report summarization. It guides you through:

1. Annotating data with a baseline DSPy program
2. Evaluating and collecting human feedback on metrics
3. Synthesizing insights from feedback
4. Optimizing the program with human-in-the-loop guidance

## Getting Started

See the [tutorial README](tutorial/README.md) for detailed instructions on running the project.

## Requirements

- Prodigy Company Plugins version 0.5.0 or later
- See `tutorial/requirements.txt` for full dependencies
