import argparse
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForMultimodalLM,
)
from tqdm import tqdm
import os
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.use_deterministic_algorithms(True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Often recommended for CUDA matmul reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--vectors_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--c", type=float, default=0.4)
    parser.add_argument("--layers", type=int, nargs=2, default=[12, 21])  # start end
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--modules", choices=["attn", "mlp", "attn+mlp"])

    args = parser.parse_args()
    start, end = args.layers

    config = AutoConfig.from_pretrained(args.model_id)

    is_multimodal = config.model_type in {
        "gemma4",
        "mistral3",
    }

    if is_multimodal:
        processor = AutoProcessor.from_pretrained(args.model_id)
        tokenizer = processor.tokenizer
        model_class = AutoModelForMultimodalLM
        layer_prefix = "model.language_model.layers"

    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model_class = AutoModelForCausalLM
        layer_prefix = "model.layers"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    vectors = torch.load(args.vectors_path)
    input_means = torch.load(args.input_path)

    if is_multimodal:
        model = model_class.from_pretrained(
            args.model_id,
            dtype="auto",
            device_map="auto",
        )
    else:
        model = model_class.from_pretrained(
            args.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    LAYERS = []

    if "attn" in args.modules:
        LAYERS += [
            f"{layer_prefix}.{i}.self_attn.o_proj"
            for i in range(start, end)
        ]

    if "mlp" in args.modules:
        LAYERS += [
            f"{layer_prefix}.{i}.mlp.down_proj"
            for i in range(start, end)
        ]

    sd = model.state_dict()

    for k in tqdm(LAYERS):
        v = vectors[k].unsqueeze(1)
        v = v / torch.linalg.vector_norm(v)

        m = sd[k + ".weight"]

        _, S, Vh = torch.linalg.svd( m.float(), full_matrices=False)

        x_mean = input_means[k].unsqueeze(1).to(m.device, dtype=m.dtype)
        
        Vh = Vh.to(m.device, dtype=m.dtype)
        dots = Vh @ x_mean

        signs = torch.sign(dots)
        signs[signs == 0] = 1

        Vh.mul_(signs)

        S = S.to(m.device, dtype=m.dtype)
        v = v.to(m.device, dtype=m.dtype)

        weight_direction = v @ (S[:, None] * Vh).sum(dim=0).unsqueeze(0)
        sd[k + ".weight"] = m + args.c * weight_direction

    model.load_state_dict(sd)

    if args.save_path is not None:
        model.save_pretrained(args.save_path)

        if is_multimodal:
            processor.save_pretrained(args.save_path)
        else:
            tokenizer.save_pretrained(args.save_path)

        print("Saved to", args.save_path)



if __name__ == "__main__":
    main()