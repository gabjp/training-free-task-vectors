import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from eval.model_utils import _is_lora, _load_and_merge_lora

from activation_steer import ActivationSteerer


TASK_GROUPS = {
    "truthfulqa": [
        "truthfulqa_mc1",
        "truthfulqa_mc2",
        "truthfulqa_gen",
    ],
    "sycophancy": [
        "sycophancy_on_nlp_survey",
        "sycophancy_on_philpapers2020",
        "sycophancy_on_political_typology_quiz",
    ],
}


TRUTHFULQA_GEN_METRICS = [
    "bleu_max,none",
    "bleu_acc,none",
    "bleu_diff,none",
    "rouge1_max,none",
    "rouge1_acc,none",
    "rouge1_diff,none",
    "rouge2_max,none",
    "rouge2_acc,none",
    "rouge2_diff,none",
    "rougeL_max,none",
    "rougeL_acc,none",
    "rougeL_diff,none",
]


PRIMARY_METRIC = {
    "mmlu": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "truthfulqa_mc1": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "truthfulqa_gen": "bleu_acc,none",
    "sycophancy_on_nlp_survey": "acc,none",
    "sycophancy_on_philpapers2020": "acc,none",
    "sycophancy_on_political_typology_quiz": "acc,none",
    "moral_stories": "acc_norm,none",
}

PRIMARY_STDERR = {
    "mmlu": "acc_stderr,none",
    "gsm8k": "exact_match_stderr,flexible-extract",
    "truthfulqa_mc1": "acc_stderr,none",
    "truthfulqa_mc2": "acc_stderr,none",
    "truthfulqa_gen": "bleu_acc_stderr,none",
    "sycophancy_on_nlp_survey": "acc_stderr,none",
    "sycophancy_on_philpapers2020": "acc_stderr,none",
    "sycophancy_on_political_typology_quiz": "acc_stderr,none",
    "moral_stories": "acc_norm_stderr,none",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model name or local path",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        choices=[
            "mmlu",
            "gsm8k",
            "truthfulqa",
            "sycophancy",
            "moral_stories",
        ],
        help="Task to evaluate. Defaults to mmlu.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for evaluation, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for lm-eval.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )

    # Eval controls
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate.",
    )
    parser.add_argument(
        "--max_gen_toks",
        type=int,
        default=1024,
        help="Maximum generated tokens for GSM8K and TruthfulQA generation.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="If set, lm-eval stores per-sample outputs. Slower/heavier.",
    )

    # Steering arguments
    parser.add_argument(
        "--steering_vector_path",
        type=str,
        default=None,
        help="Path to a .pt steering vector.",
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0,
        help="Steering coefficient.",
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1,
        help="Transformer layer index to hook. Starts at 1.",
    )
    parser.add_argument(
        "--steering_positions",
        type=str,
        default="response",
        choices=["all", "prompt", "response"],
        help="Where to apply activation steering.",
    )
    parser.add_argument(
        "--debug_steering",
        action="store_true",
        help="Print debug info from ActivationSteerer.",
    )

    parser.add_argument(
        "--attn_only",
        action="store_true",
        help="Apply steering only to self_attn.o_proj. "
            "The steering vector must be a dictionary keyed by module name.",
    )

    return parser.parse_args()


def json_safe(obj):
    if callable(obj):
        return getattr(obj, "__name__", str(obj))

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, torch.dtype):
        return str(obj)

    if hasattr(obj, "dtype"):
        try:
            return str(obj)
        except Exception:
            pass

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def load_vector(path, device, layer, attn_only=False):
    vectors = torch.load(path, map_location=device)

    if not attn_only:
        # Preserve the original behavior.
        return vectors[layer]

    # Command-line layers start at 1, but dictionary keys start at 0.
    vector_layer = layer - 1
    key_suffix = f".{vector_layer}.self_attn.o_proj"

    matching_keys = [
        key
        for key in vectors
        if isinstance(key, str) and key.endswith(key_suffix)
    ]

    if len(matching_keys) == 0:
        raise KeyError(
            f"No steering vector key ends with '{key_suffix}'. "
            f"Available keys: {list(vectors.keys())}"
        )

    if len(matching_keys) > 1:
        raise KeyError(
            f"Multiple steering vector keys end with '{key_suffix}': "
            f"{matching_keys}"
        )

    vector_key = matching_keys[0]
    print(f"Using attention steering vector: {vector_key}")

    return vectors[vector_key]


def load_lm(args):
    config = AutoConfig.from_pretrained(args.model)
    model_type = config.model_type

    steering_requested = (
        args.steering_vector_path is not None
        or args.steering_coef != 0.0
        or args.steering_layer != -1
        or args.attn_only
        or args.debug_steering
    )

    assert not (
        model_type in {"gemma4", "mistral3"} and steering_requested
    ), (
        "Inference-time activation steering is not supported for "
        f"{model_type}. Evaluate a precomputed TFTV checkpoint without "
        "passing any --steering_* arguments."
    )

    common_kwargs = {
        "pretrained": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "dtype": "bfloat16",
    }

    if model_type == "gemma4":
        # Load the full checkpoint through Gemma's native multimodal class,
        # then reuse its exact language model and LM head in a causal-LM shell.
        from accelerate import init_empty_weights
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            Gemma4ForCausalLM,
        )

        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        full_model = AutoModelForMultimodalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
        ).to(args.device)

        with init_empty_weights():
            model = Gemma4ForCausalLM(full_model.config.text_config)

        model.model = full_model.model.language_model
        model.lm_head = full_model.lm_head
        model.generation_config = full_model.generation_config
        model.config.use_cache = True

        del full_model
        torch.cuda.empty_cache()

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            device=args.device,
            batch_size=args.batch_size,
            enable_thinking=False,
        )

    elif model_type == "mistral3":
        # Current lm-eval provides a dedicated text-evaluation adapter for
        # Mistral3ForConditionalGeneration checkpoints.
        from lm_eval.models.mistral3 import Mistral3LM
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer

        lm = Mistral3LM(**common_kwargs, tie_word_embeddings=False, tokenizer=tokenizer)

    else:
        # Preserve the original Llama/Qwen loading path exactly.
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        if _is_lora(args.model):
            print("LOADING LORA")
            model = _load_and_merge_lora(args.model, torch.bfloat16, args.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                pad_token_id=tokenizer.pad_token_id,
            ).to(args.device)

        model.eval()
        model.config.use_cache = True

        if hasattr(model, "generation_config"):
            model.generation_config.do_sample = False
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id

        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )

    return lm, model_type



def run_eval(lm, args, model_type):
    eval_kwargs = {
        "model": lm,
        "tasks": [args.task],
        "batch_size": args.batch_size,
        "device": args.device,
        "num_fewshot": 0,
        "limit": args.limit,
        "log_samples": args.log_samples,
    }

    # Preserve the original protocol:
    # MMLU and the other likelihood tasks use raw completion prompts.
    # Only GSM8K uses the model's chat template.
    if args.task == "gsm8k":
        eval_kwargs["apply_chat_template"] = True

    if args.task in ["gsm8k", "truthfulqa"]:
        eval_kwargs["gen_kwargs"] = {
            "do_sample": False,
            "max_gen_toks": args.max_gen_toks,
        }

    if args.task == "mmlu" and model_type == "gemma4":
        print("enabling few_shot_as_multiturn")
        eval_kwargs["fewshot_as_multiturn"] = True
        eval_kwargs["apply_chat_template"] = True
        eval_kwargs["num_fewshot"] = 5
    

    return simple_evaluate(**eval_kwargs)


def print_task_results(results, task):
    tasks_to_print = TASK_GROUPS.get(task, [task])

    for current_task in tasks_to_print:
        if current_task not in results["results"]:
            print(f"\nResults for task: {current_task}")
            print("NOT FOUND")
            continue

        task_results = results["results"][current_task]

        print(f"\nResults for task: {current_task}")

        if current_task == "truthfulqa_gen":
            for metric_key in TRUTHFULQA_GEN_METRICS:
                stderr_key = metric_key.replace(",none", "_stderr,none")

                print(metric_key + ":")
                print(task_results.get(metric_key, "NOT FOUND"))

                print(stderr_key + ":")
                print(task_results.get(stderr_key, "NOT FOUND"))

            continue

        metric_key = PRIMARY_METRIC[current_task]
        stderr_key = PRIMARY_STDERR[current_task]

        print(metric_key + ":")
        print(task_results.get(metric_key, "NOT FOUND"))

        print(stderr_key + ":")
        print(task_results.get(stderr_key, "NOT FOUND"))

        if current_task == "gsm8k":
            strict_key = "exact_match,strict-match"
            strict_stderr_key = "exact_match_stderr,strict-match"

            if strict_key in task_results:
                print(strict_key + ":")
                print(task_results[strict_key])

            if strict_stderr_key in task_results:
                print(strict_stderr_key + ":")
                print(task_results[strict_stderr_key])


def main():
    args = parse_args()
    print(args)

    lm, model_type = load_lm(args)
    model = lm.model


    steering_info = None

    if args.steering_vector_path is not None and args.steering_coef != 0.0:
        if args.steering_layer < 1:
            raise ValueError(
                "--steering_layer should start at 1 when using activation steering."
            )

        steering_vector = load_vector(
            args.steering_vector_path,
            device=args.device,
            layer=args.steering_layer,
            attn_only=args.attn_only
        )

        steering_info = {
            "vector_path": args.steering_vector_path,
            "coef": args.steering_coef,
            "layer": args.steering_layer,
            "positions": args.steering_positions,
            "attn_only": args.attn_only,
        }

        with ActivationSteerer(
            model,
            steering_vector=steering_vector,
            coeff=args.steering_coef,
            layer_idx=args.steering_layer - 1,
            positions=args.steering_positions,
            attn_only=args.attn_only,
            debug=args.debug_steering,
        ):
            results = run_eval(lm, args, model_type)
    else:
        results = run_eval(lm, args, model_type)

    print_task_results(results, args.task)

    if args.task in TASK_GROUPS:
        primary_metric = {
            task_name: PRIMARY_METRIC[task_name]
            for task_name in TASK_GROUPS[args.task]
        }
    else:
        primary_metric = PRIMARY_METRIC[args.task]

    results["eval_config"] = {
        "task": args.task,
        "num_fewshot": 0,
        "batch_size": args.batch_size,
        "device": args.device,
        "limit": args.limit,
        "max_gen_toks": (
            args.max_gen_toks
            if args.task in ["gsm8k", "truthfulqa"]
            else None
        ),
        "primary_metric": primary_metric,
        "log_samples": args.log_samples,
        "apply_chat_template": args.task == "gsm8k",
        "model_type": model_type,
    }

    if steering_info is not None:
        results["steering"] = steering_info

    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(json_safe(results), f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()