#!/usr/bin/env python3

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    pretrained = load_model(args.pretrained_model)

    merged = PeftModel.from_pretrained(
        pretrained,
        args.lora_path,
    ).merge_and_unload()

    initialization = load_model(args.pretrained_model)
    initialization_params = dict(initialization.named_parameters())

    # reversed = initialization - (merged - initialization)
    #          = 2 * initialization - merged
    with torch.no_grad():
        for name, merged_param in merged.named_parameters():
            initial_param = initialization_params[name]

            reversed_param = (
                2.0 * initial_param.float()
                - merged_param.float()
            )

            merged_param.copy_(
                reversed_param.to(merged_param.dtype)
            )

    merged.save_pretrained(
        args.save_path,
        safe_serialization=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.save_pretrained(args.save_path)

    print(f"Reversed model saved to {args.save_path}")


if __name__ == "__main__":
    main()