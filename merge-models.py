import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="1 to 3 finetuned models")
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--merge_mode",
        choices=["sum", "avg"],
        default="sum",
        help="Whether to sum or average the task vectors",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (1 <= len(args.models) <= 3):
        raise ValueError("Provide between 1 and 3 models in --models")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
    )
    base_state = base_model.state_dict()

    diff_states = []

    for model_path in args.models:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        model_state = model.state_dict()

        diff = {}
        for k in base_state:
            diff[k] = model_state[k] - base_state[k]

        diff_states.append(diff)
        del model

    print("Merging differences...")
    merged_diff = {}
    for k in base_state:
        merged_diff[k] = sum(diff[k] for diff in diff_states)
        if args.merge_mode == "avg":
            merged_diff[k] = merged_diff[k] / len(diff_states)

    print("Applying merged difference to base model...")
    new_state = {}
    for k in base_state:
        new_state[k] = base_state[k] + merged_diff[k]

    base_model.load_state_dict(new_state)

    print("Saving merged model and tokenizer...")
    os.makedirs(args.output_path, exist_ok=True)
    base_model.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)

    print(f"Saved merged model to: {args.output_path}")

if __name__ == "__main__":
    main()