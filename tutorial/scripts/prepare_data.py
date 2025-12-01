"""
Preprocess MultiClinSUM data for the Prodigy DSPy tutorial.

This script loads clinical notes and summaries from the downloaded MultiClinSUM
dataset and converts them into Prodigy JSONL format, split deterministically into
dev and test sets using hash-based splitting.

Usage:
    python scripts/prepare_data.py --input-path data/multiclinsum_gs_train_en \
        --dev-path assets/clinical_notes_dev.jsonl \
        --test-path assets/clinical_notes_test.jsonl \
        --n-dev 30 --n-test 100
"""

from typing import List, Dict, Any, Tuple
import typer
from pathlib import Path
import srsly
from wasabi import msg
import os
import hashlib


def load_multiclinsum_examples(
    dataset_path: Path, sample_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Load clinical notes and summaries from MultiClinSUM dataset.

    Args:
        dataset_path: Path to the extracted multiclinsum_gs_train_en directory
        sample_size: Number of examples to load

    Returns:
        List of examples in Prodigy format with gold_summary in meta
    """
    fulltext_path = dataset_path / "fulltext"
    summary_path = dataset_path / "summaries"

    if not fulltext_path.exists() or not summary_path.exists():
        msg.fail(
            f"Expected directories not found in {dataset_path}",
            f"fulltext exists: {fulltext_path.exists()}",
            f"summaries exists: {summary_path.exists()}",
        )
        return []

    fulltext_files = sorted(
        [f for f in os.listdir(fulltext_path) if f.endswith(".txt")]
    )
    msg.info(f"Found {len(fulltext_files)} examples in dataset")

    if len(fulltext_files) > sample_size:
        fulltext_files = fulltext_files[:sample_size]
        msg.info(f"Using first {sample_size} examples")

    examples = []
    skipped = 0

    for text_filename in fulltext_files:
        summary_filename = text_filename.replace(".txt", "_sum.txt")
        text_filepath = fulltext_path / text_filename
        summary_filepath = summary_path / summary_filename

        if not summary_filepath.exists():
            skipped += 1
            continue

        try:
            with open(text_filepath, "r", encoding="utf-8") as f:
                document_text = f.read()
            with open(summary_filepath, "r", encoding="utf-8") as f:
                gold_summary = f.read()

            # Convert to Prodigy format
            # The 'text' field contains the clinical note
            # gold_summary is included here but will be removed from dev set later
            example = {
                "text": document_text,
                "gold_summary": gold_summary,
                "meta": {
                    "source": "MultiClinSUM",
                    "filename": text_filename,
                },
            }
            examples.append(example)

        except Exception as e:
            msg.warn(f"Skipping file {text_filename}: {e}")
            skipped += 1

    if skipped > 0:
        msg.warn(f"Skipped {skipped} files")

    msg.good(f"Loaded {len(examples)} examples")
    return examples


def print_dataset_stats(examples: List[Dict[str, Any]], description: str) -> None:
    """
    Print statistics about the dataset.

    Args:
        examples: List of examples
        description: Description of the dataset
    """
    if not examples:
        return

    text_lengths = [len(ex.get("text", "").split()) for ex in examples]
    summary_lengths = [len(ex.get("gold_summary", "").split()) for ex in examples]

    avg_text_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    avg_summary_len = (
        sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
    )

    print(f"\n{description} Statistics:")
    print("-" * 50)
    print(f"  Total examples: {len(examples)}")
    print(f"  Avg text length: {avg_text_len:.0f} words")
    print(f"  Avg summary length: {avg_summary_len:.0f} words")
    if avg_summary_len > 0:
        print(f"  Compression ratio: {avg_text_len / avg_summary_len:.1f}x")


def hash_split(
    examples: List[Dict[str, Any]], n_dev: int, n_test: int, split_key: str = "text"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    # Assign a deterministic score to each example
    scored = []
    for ex in examples:
        content = ex.get(split_key, "")
        h = int(hashlib.md5(content.encode()).hexdigest(), 16)
        scored.append((h, ex))

    # Sort by hash deterministically
    scored.sort(key=lambda x: x[0])

    # Split
    dev = [ex for _, ex in scored[:n_dev]]
    test = [ex for _, ex in scored[n_dev : n_dev + n_test]]

    return dev, test


def main(
    input_path: Path = typer.Option(
        "data/multiclinsum_gs_train_en",
        "--input-path",
        "-i",
        help="Path to the extracted MultiClinSUM dataset directory",
    ),
    dev_path: Path = typer.Option(
        "assets/clinical_notes_dev.jsonl",
        "--dev-path",
        "-d",
        help="Path to save development set JSONL",
    ),
    test_path: Path = typer.Option(
        "assets/clinical_notes_test.jsonl",
        "--test-path",
        "-t",
        help="Path to save test set JSONL",
    ),
    n_dev: int = typer.Option(30, "--n-dev", help="Number of examples for dev set"),
    n_test: int = typer.Option(20, "--n-test", help="Number of examples for test set"),
) -> None:
    """
    Preprocess MultiClinSUM data into Prodigy format.

    Loads clinical notes and summaries from the MultiClinSUM dataset,
    splits them deterministically using hash-based splitting, and saves as JSONL files.
    """
    msg.divider("MultiClinSUM Data Preprocessing")

    input_path = Path(input_path)
    if not input_path.exists():
        msg.fail(f"Input directory not found: {input_path}")
        raise typer.Exit(1)

    msg.info(f"Loading examples from {input_path}...")
    total_needed = n_dev + n_test
    all_examples = load_multiclinsum_examples(input_path, sample_size=total_needed)

    if not all_examples:
        msg.fail("No examples loaded from dataset")
        raise typer.Exit(1)

    msg.info(f"Splitting into dev ({n_dev}) and test ({n_test}) sets...")
    dev_examples, test_examples = hash_split(all_examples, n_dev, n_test)

    # Remove gold_summary from dev set (users will add it via annotation)
    for ex in dev_examples:
        ex["meta"].pop("gold_summary", None)

    print_dataset_stats(dev_examples, "Dev Set")
    print_dataset_stats(test_examples, "Test Set")

    dev_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    msg.info(f"Saving dev set to {dev_path}...")
    srsly.write_jsonl(dev_path, dev_examples)

    msg.info(f"Saving test set to {test_path}...")
    srsly.write_jsonl(test_path, test_examples)

    msg.divider("✓ Complete")
    print(f"\nDev set:  {dev_path} ({len(dev_examples)} examples)")
    print(f"Test set: {test_path} ({len(test_examples)} examples)")


if __name__ == "__main__":
    typer.run(main)
