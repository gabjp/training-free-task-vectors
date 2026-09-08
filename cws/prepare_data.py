#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def write_axolotl_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": str(row["question"]),
                    },
                    {
                        "role": "assistant",
                        "content": str(row["answer"]),
                    },
                ]
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def convert_csvs(
    positive_path: str,
    negative_path: str,
    trait: str,
    output_dir: str | None = None,
    threshold: float = 50,
):
    positive_path = Path(positive_path)
    negative_path = Path(negative_path)

    persona_pos = pd.read_csv(positive_path)
    persona_neg = pd.read_csv(negative_path)

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------
    required_columns = {"question", "answer", "coherence", trait}

    for name, df in [
        ("positive", persona_pos),
        ("negative", persona_neg),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} CSV is missing columns: {sorted(missing)}"
            )

    # ---------------------------------------------------------
    # Make sure both CSVs are aligned row-by-row
    # ---------------------------------------------------------
    if len(persona_pos) != len(persona_neg):
        raise ValueError(
            "Positive and negative CSVs have different numbers of rows: "
            f"{len(persona_pos)} vs {len(persona_neg)}"
        )

    if not persona_pos["question"].equals(persona_neg["question"]):
        raise ValueError(
            "Questions are not aligned between the positive and negative CSVs."
        )

    # ---------------------------------------------------------
    # Joint filtering mask
    # ---------------------------------------------------------
    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )

    filtered_pos = persona_pos[mask].reset_index(drop=True)
    filtered_neg = persona_neg[mask].reset_index(drop=True)

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------
    if output_dir is None:
        output_dir = positive_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    positive_output = output_dir / f"{trait}_pos_axolotl.jsonl"
    negative_output = output_dir / f"{trait}_neg_axolotl.jsonl"

    # ---------------------------------------------------------
    # Save separately
    # ---------------------------------------------------------
    write_axolotl_jsonl(filtered_pos, positive_output)
    write_axolotl_jsonl(filtered_neg, negative_output)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    print("=" * 70)
    print(f"Trait:               {trait}")
    print(f"Threshold:           {threshold}")
    print()
    print(f"Positive CSV:        {positive_path}")
    print(f"Negative CSV:        {negative_path}")
    print()
    print(f"Original questions:  {len(persona_pos)}")
    print(f"Kept questions:      {mask.sum()}")
    print(f"Removed questions:   {(~mask).sum()}")
    print()
    print(f"Positive examples:   {len(filtered_pos)}")
    print(f"Negative examples:   {len(filtered_neg)}")
    print()
    print(f"Positive output:     {positive_output}")
    print(f"Negative output:     {negative_output}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Jointly filter positive/negative persona CSVs and save "
            "them as separate Axolotl JSONL datasets."
        )
    )

    parser.add_argument(
        "positive_csv",
        help="CSV containing positive-trait answers",
    )

    parser.add_argument(
        "negative_csv",
        help="CSV containing negative-trait answers",
    )

    parser.add_argument(
        "--trait",
        required=True,
        help="Trait column name, e.g. evil, sycophantic, hallucinating",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Directory where the Axolotl JSONL files will be saved",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=50,
        help="Trait threshold (default: 50)",
    )

    args = parser.parse_args()

    convert_csvs(
        positive_path=args.positive_csv,
        negative_path=args.negative_csv,
        trait=args.trait,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()