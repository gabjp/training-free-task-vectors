#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


TRAIT_COLUMNS = ["evil", "sycophantic", "hallucinating"]


def find_trait_column(df: pd.DataFrame) -> str:
    """Find which trait score is present in the CSV."""
    found = [trait for trait in TRAIT_COLUMNS if trait in df.columns]

    if len(found) == 0:
        raise ValueError(
            f"Could not find a trait column. "
            f"Expected one of: {TRAIT_COLUMNS}"
        )

    if len(found) > 1:
        raise ValueError(
            f"Found multiple trait columns: {found}. "
            "Expected exactly one."
        )

    return found[0]


def analyze_csv(csv_path: Path) -> dict:
    """Analyze one CSV and return all statistics."""

    df = pd.read_csv(csv_path)

    if "prompt" not in df.columns:
        raise ValueError("CSV must contain a 'prompt' column.")

    if "coherence" not in df.columns:
        raise ValueError("CSV must contain a 'coherence' column.")

    trait = find_trait_column(df)

    df[trait] = pd.to_numeric(df[trait], errors="raise")
    df["coherence"] = pd.to_numeric(
        df["coherence"], errors="raise"
    )

    score_columns = [trait, "coherence"]

    prompt_counts = df.groupby("prompt").size()

    if prompt_counts.nunique() != 1:
        raise ValueError(
            "Not every prompt has the same number of generations."
        )

    n_prompts = df["prompt"].nunique()
    n_generations_per_prompt = int(prompt_counts.iloc[0])

    overall_means = df[score_columns].mean()

    generation_std = df[score_columns].std(ddof=1)

    df = df.copy()
    df["run"] = df.groupby("prompt").cumcount() + 1

    run_means = df.groupby("run")[score_columns].mean()
    run_std = run_means.std(ddof=1)

    prompt_means = df.groupby("prompt")[score_columns].mean()
    prompt_std = prompt_means.std(ddof=1)

    return {
        "file": csv_path.name,
        "trait": trait,
        "n_prompts": n_prompts,
        "n_generations_per_prompt": n_generations_per_prompt,
        "n_total": len(df),

        "score": overall_means[trait],
        "score_std_generation": generation_std[trait],
        "score_std_run": run_std[trait],
        "score_std_prompt": prompt_std[trait],

        "coherence": overall_means["coherence"],
        "coherence_std_generation": generation_std["coherence"],
        "coherence_std_run": run_std["coherence"],
        "coherence_std_prompt": prompt_std["coherence"],
    }


def print_single_result(result: dict, csv_path: Path) -> None:
    """Print detailed results for a single CSV."""

    print("=" * 70)
    print(f"File: {csv_path}")
    print(f"Trait: {result['trait']}")
    print(f"Number of prompts: {result['n_prompts']:.2f}")
    print(
        f"Generations per prompt: "
        f"{result['n_generations_per_prompt']:.2f}"
    )
    print(f"Total generations: {result['n_total']:.2f}")
    print("=" * 70)

    print("\nOVERALL AVERAGES")
    print("-" * 70)
    print(
        f"Average {result['trait']:15s}: "
        f"{result['score']:.2f}"
    )
    print(
        f"Average coherence     : "
        f"{result['coherence']:.2f}"
    )

    print("\nSTD OVER GENERATIONS (NO AGGREGATION)")
    print("-" * 70)
    print(
        f"{result['trait']:20s}: "
        f"{result['score_std_generation']:.2f}"
    )
    print(
        f"{'coherence':20s}: "
        f"{result['coherence_std_generation']:.2f}"
    )

    print("\nSTD OVER RUNS")
    print("-" * 70)
    print(
        f"{result['trait']:20s}: "
        f"{result['score_std_run']:.2f}"
    )
    print(
        f"{'coherence':20s}: "
        f"{result['coherence_std_run']:.2f}"
    )

    print("\nSTD OVER PROMPTS")
    print("-" * 70)
    print(
        f"{result['trait']:20s}: "
        f"{result['score_std_prompt']:.2f}"
    )
    print(
        f"{'coherence':20s}: "
        f"{result['coherence_std_prompt']:.2f}"
    )


def print_folder_results(results: list[dict], folder: Path) -> None:
    """Print results from multiple CSVs as one organized table."""

    print("=" * 130)
    print(f"Folder: {folder}")
    print(f"CSV files analyzed: {len(results):.2f}")
    print("=" * 130)

    rows = []

    for result in results:
        rows.append({
            "File": result["file"],
            "Trait": result["trait"].capitalize(),
            "Score": f"{result['score']:.2f}",
            "Score std (gen - run - prompt)": (
                f"{result['score_std_generation']:.2f} - "
                f"{result['score_std_run']:.2f} - "
                f"{result['score_std_prompt']:.2f}"
            ),
            "Coherence": f"{result['coherence']:.2f}",
            "Coherence std (gen - run - prompt)": (
                f"{result['coherence_std_generation']:.2f} - "
                f"{result['coherence_std_run']:.2f} - "
                f"{result['coherence_std_prompt']:.2f}"
            ),
        })

    summary = pd.DataFrame(rows)

    print()
    print(summary.to_string(index=False))
    print()

    print("=" * 130)
    print("TAB-SEPARATED VERSION")
    print("=" * 130)
    print(summary.to_csv(sep="\t", index=False).strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute trait/coherence statistics for either one CSV "
            "or every CSV in a folder."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a CSV file or a folder containing CSV files.",
    )

    args = parser.parse_args()
    path = args.path

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a .csv file, got: {path}"
            )

        result = analyze_csv(path)
        print_single_result(result, path)

    elif path.is_dir():
        csv_files = sorted(path.glob("*.csv"))

        if not csv_files:
            raise ValueError(
                f"No .csv files found inside: {path}"
            )

        results = []
        failed = []

        for csv_path in csv_files:
            try:
                result = analyze_csv(csv_path)
                results.append(result)

            except Exception as exc:
                failed.append((csv_path.name, str(exc)))

        if results:
            print_folder_results(results, path)

        if failed:
            print()
            print("=" * 70)
            print("FILES THAT COULD NOT BE ANALYZED")
            print("=" * 70)

            for filename, error in failed:
                print(f"{filename}: {error}")

    else:
        raise FileNotFoundError(
            f"Path does not exist: {path}"
        )


if __name__ == "__main__":
    main()