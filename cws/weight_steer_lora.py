#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32

    raise ValueError(f"Unsupported dtype: {name}")


def validate_adapter(adapter_path: str):
    adapter_path = Path(adapter_path)

    config_path = adapter_path / "adapter_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_path}.\n"
            "Pass the Axolotl LoRA output directory containing "
            "adapter_config.json and adapter_model.safetensors."
        )

    config = PeftConfig.from_pretrained(adapter_path)

    print(f"Adapter: {adapter_path}")
    print(f"  PEFT type:  {config.peft_type}")
    print(f"  Base model: {config.base_model_name_or_path}")


def load_base(base_model: str, dtype):
    return AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )


def load_and_merge_adapter(
    base_model: str,
    adapter_path: str,
    dtype,
):
    """
    Load:

        base + LoRA

    and merge the LoRA into the base weights entirely in memory.

    Nothing is saved.
    """

    print(f"\nLoading fresh base model for adapter:")
    print(f"  {adapter_path}")

    model = load_base(base_model, dtype)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    return model


def tensor_storage_id(tensor: torch.Tensor):
    """
    Identify an exact tensor/storage view.

    This prevents applying arithmetic twice when two state-dict keys
    point to the same tied parameter.
    """
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def make_difference_in_place(
    positive_model,
    negative_model,
):
    """
    Transform positive_model in place:

        positive_model <- positive_model - negative_model

    After this function, positive_model stores the contrastive
    weight direction P - N.
    """

    positive_state = positive_model.state_dict()
    negative_state = negative_model.state_dict()

    if set(positive_state) != set(negative_state):
        missing = set(positive_state) - set(negative_state)
        extra = set(negative_state) - set(positive_state)

        raise ValueError(
            "Positive and negative merged models have different "
            "state-dict structures.\n"
            f"Missing from negative: {sorted(missing)[:10]}\n"
            f"Extra in negative: {sorted(extra)[:10]}"
        )

    processed_storages = set()
    updated = 0

    with torch.no_grad():
        for i, key in enumerate(positive_state):
            pos = positive_state[key]
            neg = negative_state[key]

            if pos.shape != neg.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{pos.shape} vs {neg.shape}"
                )

            if pos.dtype in (torch.int64, torch.uint8):
                continue

            storage_id = tensor_storage_id(pos)


            if storage_id in processed_storages:
                continue

            processed_storages.add(storage_id)

            pos.sub_(neg)

            updated += 1

            if updated % 50 == 0:
                print(f"  Computed difference for {updated} tensors")

    print(f"Computed P - N for {updated} tensors.")


def apply_difference_to_base(
    base_model,
    difference_model,
    coefficient: float,
):
    """
    Transform base_model into:

        B + coefficient * (P - N)
    """

    base_state = base_model.state_dict()
    difference_state = difference_model.state_dict()

    if set(base_state) != set(difference_state):
        raise ValueError(
            "Base model and steering direction have different "
            "state-dict structures."
        )

    processed_storages = set()
    updated = 0

    with torch.no_grad():
        for key in base_state:
            base = base_state[key]
            delta = difference_state[key]

            if base.dtype in (torch.int64, torch.uint8):
                continue

            if base.shape != delta.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{base.shape} vs {delta.shape}"
                )

            storage_id = tensor_storage_id(base)

            if storage_id in processed_storages:
                continue

            processed_storages.add(storage_id)

            base.add_(
                delta,
                alpha=coefficient,
            )

            updated += 1

            if updated % 50 == 0:
                print(f"  Applied steering to {updated} tensors")

    print(f"Applied steering to {updated} tensors.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a contrastive weight-steered model directly "
            "from two LoRA adapters."
        )
    )

    parser.add_argument(
        "--base_model",
        required=True,
        help="Base Hugging Face model.",
    )

    parser.add_argument(
        "--positive_adapter",
        required=True,
        help="LoRA adapter trained on positive-trait responses.",
    )

    parser.add_argument(
        "--negative_adapter",
        required=True,
        help="LoRA adapter trained on negative-trait responses.",
    )

    parser.add_argument(
        "--coefficient",
        "-k",
        type=float,
        required=True,
        help="Weight-steering coefficient.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for the final weight-steered model.",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "auto",
            "bfloat16",
            "float16",
            "float32",
        ],
        default="bfloat16",
        help="Model dtype used during merging/arithmetic.",
    )

    args = parser.parse_args()

    dtype = get_dtype(args.dtype)

    print("=" * 70)
    print("Contrastive Weight Steering")
    print("=" * 70)
    print(f"Base:             {args.base_model}")
    print(f"Positive adapter: {args.positive_adapter}")
    print(f"Negative adapter: {args.negative_adapter}")
    print(f"Coefficient:      {args.coefficient}")
    print(f"Dtype:            {args.dtype}")
    print(f"Output:           {args.output_dir}")
    print()
    print(
        "Formula: B + k * (P - N)"
    )
    print("=" * 70)

    validate_adapter(args.positive_adapter)
    validate_adapter(args.negative_adapter)

    print("\n" + "=" * 70)
    print("STEP 1: Merge positive LoRA")
    print("=" * 70)

    positive_model = load_and_merge_adapter(
        args.base_model,
        args.positive_adapter,
        dtype,
    )

    print("\n" + "=" * 70)
    print("STEP 2: Merge negative LoRA")
    print("=" * 70)

    negative_model = load_and_merge_adapter(
        args.base_model,
        args.negative_adapter,
        dtype,
    )

    print("\n" + "=" * 70)
    print("STEP 3: Compute contrastive direction P - N")
    print("=" * 70)

    make_difference_in_place(
        positive_model,
        negative_model,
    )

    # negative_model is no longer needed.
    del negative_model
    gc.collect()

    print("\n" + "=" * 70)
    print("STEP 4: Reload clean base model")
    print("=" * 70)

    steered_model = load_base(
        args.base_model,
        dtype,
    )

    print("\n" + "=" * 70)
    print("STEP 5: Apply steering direction")
    print("=" * 70)

    apply_difference_to_base(
        steered_model,
        positive_model,
        args.coefficient,
    )

    # Difference no longer needed.
    del positive_model
    gc.collect()


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 6: Save final model")
    print("=" * 70)

    steered_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.base_model,
        "positive_adapter": args.positive_adapter,
        "negative_adapter": args.negative_adapter,
        "coefficient": args.coefficient,
        "dtype": args.dtype,
        "formula": (
            "theta_steered = theta_base + "
            "k * (theta_positive - theta_negative)"
        ),
    }

    with open(
        output_dir / "weight_steering_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Final model saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()