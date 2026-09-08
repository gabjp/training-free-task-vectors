from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import json
import torch
import os
import argparse
import torch.nn as nn
import pandas as pd


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)

    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    pos_prompts = persona_pos_effective["prompt"].tolist()
    neg_prompts = persona_neg_effective["prompt"].tolist()

    pos_responses = persona_pos_effective["answer"].tolist()
    neg_responses = persona_neg_effective["answer"].tolist()

    return (
        persona_pos_effective,
        persona_neg_effective,
        pos_prompts,
        neg_prompts,
        pos_responses,
        neg_responses,
    )


class LinearStatsCollector:
    """
    Collects:
      - output mean over prompt tokens per example  -> sum_out_prompt[name] (out_features)
      - output mean over response tokens per example-> sum_out_resp[name]   (out_features)
      - input mean over ALL tokens per example      -> sum_in_all[name]     (in_features)

    Aggregation is "mean over tokens, then mean over examples".
    """

    def __init__(self, linear_modules: dict[str, nn.Linear], include_lm_head: bool = True):
        self.linear_modules = linear_modules
        self.module_names = list(linear_modules.keys())

        # running sums (CPU) and counts
        self.sum_out_prompt = {}
        self.sum_out_resp = {}
        self.sum_in_all = {}
        self.n_examples = 0

        # context set before each forward
        self.current_prompt_len = None

        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, mod in self.linear_modules.items():
            self.hooks.append(mod.register_forward_hook(self._make_hook(name)))

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @staticmethod
    def _ensure_2d_or_3d(x: torch.Tensor) -> torch.Tensor:
        # Linear layers in HF typically see [B,T,C] or [N,C].
        if x.dim() == 2:
            # [N, C] -> treat N as "tokens"
            return x.unsqueeze(0)  # [1, N, C]
        if x.dim() == 3:
            return x
        # fallback: flatten everything except last dim as tokens
        return x.reshape(1, -1, x.shape[-1])

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            # inputs[0] is the tensor input to Linear
            x_in = inputs[0]
            y_out = output

            if not torch.is_tensor(x_in) or not torch.is_tensor(y_out):
                return

            x_in = self._ensure_2d_or_3d(x_in)
            y_out = self._ensure_2d_or_3d(y_out)

            # x_in: [B, T, in_features], y_out: [B, T, out_features]
            # We run with B=1 in this script; keep generic anyway.
            B, T, _ = y_out.shape
            prompt_len = int(self.current_prompt_len) if self.current_prompt_len is not None else 0
            prompt_len = max(0, min(prompt_len, T))

            # Mean over ALL tokens for inputs
            in_mean_all = x_in.mean(dim=1).mean(dim=0)  # [in_features]

            # Mean over prompt / response tokens for outputs
            if prompt_len > 0:
                out_mean_prompt = y_out[:, :prompt_len, :].mean(dim=1).mean(dim=0)  # [out_features]
            else:
                out_mean_prompt = None

            if prompt_len < T:
                out_mean_resp = y_out[:, prompt_len:, :].mean(dim=1).mean(dim=0)  # [out_features]
            else:
                out_mean_resp = None

            # Accumulate on CPU (float64 for stability, cast later)
            in_mean_all_cpu = in_mean_all.detach().to("cpu", dtype=torch.float64)
            if name not in self.sum_in_all:
                self.sum_in_all[name] = torch.zeros_like(in_mean_all_cpu)
            self.sum_in_all[name] += in_mean_all_cpu

            if out_mean_prompt is not None:
                out_mean_prompt_cpu = out_mean_prompt.detach().to("cpu", dtype=torch.float64)
                if name not in self.sum_out_prompt:
                    self.sum_out_prompt[name] = torch.zeros_like(out_mean_prompt_cpu)
                self.sum_out_prompt[name] += out_mean_prompt_cpu

            if out_mean_resp is not None:
                out_mean_resp_cpu = out_mean_resp.detach().to("cpu", dtype=torch.float64)
                if name not in self.sum_out_resp:
                    self.sum_out_resp[name] = torch.zeros_like(out_mean_resp_cpu)
                self.sum_out_resp[name] += out_mean_resp_cpu

        return hook

    def step_example(self, prompt_len: int):
        self.current_prompt_len = prompt_len
        self.n_examples += 1

    def finalize_means(self):
        """
        Returns dicts:
          out_prompt_mean[name] : float32 CPU
          out_resp_mean[name]   : float32 CPU
          in_all_mean[name]     : float32 CPU
        Any missing (e.g. prompt_len==0 always) keys will be absent.
        """
        if self.n_examples == 0:
            raise RuntimeError("No examples processed.")

        out_prompt_mean = {}
        out_resp_mean = {}
        in_all_mean = {}

        for name, s in self.sum_in_all.items():
            in_all_mean[name] = (s / self.n_examples).to(dtype=torch.float32)

        for name, s in self.sum_out_prompt.items():
            out_prompt_mean[name] = (s / self.n_examples).to(dtype=torch.float32)

        for name, s in self.sum_out_resp.items():
            out_resp_mean[name] = (s / self.n_examples).to(dtype=torch.float32)

        return out_prompt_mean, out_resp_mean, in_all_mean


def collect_linear_means(model, tokenizer, prompts, responses, include_lm_head: bool = True):
    """
    Runs the model on each (prompt+response) and collects:
      - output means per linear layer for prompt tokens and response tokens
      - input means per linear layer for all tokens

    Returns:
      out_prompt_mean, out_resp_mean, in_all_mean, n_examples, module_names
    """
    # collect linear layers (optionally include lm_head)
    linear_modules = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if (not include_lm_head) and (name == "lm_head" or name.endswith(".lm_head")):
                continue
            linear_modules[name] = mod

    collector = LinearStatsCollector(linear_modules, include_lm_head=include_lm_head)

    if hasattr(model.model, "language_model"):
        device = model.model.language_model.embed_tokens.weight.device
    else:
        device = model.model.embed_tokens.weight.device

    texts = [p + a for p, a in zip(prompts, responses)]

    for text, prompt in tqdm(list(zip(texts, prompts)), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        collector.step_example(prompt_len)

        with torch.inference_mode():
            # forward pass triggers hooks
            _ = model.model(**inputs, use_cache=False)

        del inputs

    out_prompt_mean, out_resp_mean, in_all_mean = collector.finalize_means()
    n_examples = collector.n_examples
    module_names = collector.module_names
    collector.close()

    return out_prompt_mean, out_resp_mean, in_all_mean, n_examples, module_names


def save_persona_vector_linear(
    model_name,
    pos_path,
    neg_path,
    trait,
    save_dir,
    threshold=50,
    include_lm_head: bool = True,
):
    config = AutoConfig.from_pretrained(model_name)

    if config.model_type in {"gemma4", "mistral3"}:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
        )
        tokenizer = AutoProcessor.from_pretrained(model_name).tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "22GiB"},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    (
        _pos_df,
        _neg_df,
        pos_prompts,
        neg_prompts,
        pos_responses,
        neg_responses,
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    # Collect means for pos / neg
    pos_out_prompt, pos_out_resp, pos_in_all, n_pos, module_names_pos = collect_linear_means(
        model, tokenizer, pos_prompts, pos_responses, include_lm_head=include_lm_head
    )
    neg_out_prompt, neg_out_resp, neg_in_all, n_neg, module_names_neg = collect_linear_means(
        model, tokenizer, neg_prompts, neg_responses, include_lm_head=include_lm_head
    )

    # sanity: module lists should match (same model)
    if module_names_pos != module_names_neg:
        # still proceed by intersecting keys
        common = sorted(set(module_names_pos).intersection(set(module_names_neg)))
    else:
        common = module_names_pos

    # Persona vectors at OUTPUT of every linear layer (pos - neg)
    # We compute for prompt and response separately (like your original code).
    persona_out_prompt_diff = {}
    persona_out_resp_diff = {}

    for name in common:
        if name in pos_out_prompt and name in neg_out_prompt:
            persona_out_prompt_diff[name] = (pos_out_prompt[name] - neg_out_prompt[name]).contiguous()
        if name in pos_out_resp and name in neg_out_resp:
            persona_out_resp_diff[name] = (pos_out_resp[name] - neg_out_resp[name]).contiguous()

    # INPUT means for every linear layer: pos, neg, both
    in_mean_pos = {}
    in_mean_neg = {}
    in_mean_both = {}

    for name in common:
        if name in pos_in_all and name in neg_in_all:
            in_mean_pos[name] = pos_in_all[name].contiguous()
            in_mean_neg[name] = neg_in_all[name].contiguous()
            # "both": mean across examples from both sets (weighted by counts)
            in_mean_both[name] = (
                (
                    pos_in_all[name] * n_pos
                    + neg_in_all[name] * n_neg
                )
                / (n_pos + n_neg)
            ).contiguous()

    os.makedirs(save_dir, exist_ok=True)

    # Save persona vectors (outputs)
    torch.save(
        persona_out_prompt_diff,
        os.path.join(save_dir, f"{trait}_linear_out_prompt_mean_diff.pt"),
    )
    torch.save(
        persona_out_resp_diff,
        os.path.join(save_dir, f"{trait}_linear_out_response_mean_diff.pt"),
    )

    # Save input means
    torch.save(
        in_mean_pos,
        os.path.join(save_dir, f"{trait}_linear_in_mean_pos.pt"),
    )
    torch.save(
        in_mean_neg,
        os.path.join(save_dir, f"{trait}_linear_in_mean_neg.pt"),
    )
    torch.save(
        in_mean_both,
        os.path.join(save_dir, f"{trait}_linear_in_mean_both.pt"),
    )

    # Save module ordering for reproducibility
    torch.save(
        common,
        os.path.join(save_dir, f"{trait}_linear_module_names.pt"),
    )

    print(f"Saved linear-layer persona vectors + input means to {save_dir}")
    print(f"- #pos examples: {n_pos}, #neg examples: {n_neg}")
    print(f"- #linear modules saved: {len(common)}")
    print(f"- include_lm_head = {include_lm_head}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument(
        "--include_lm_head",
        action="store_true",
        help="Also hook lm_head if it is a Linear layer.",
    )
    args = parser.parse_args()

    save_persona_vector_linear(
        args.model_name,
        args.pos_path,
        args.neg_path,
        args.trait,
        args.save_dir,
        threshold=args.threshold,
        include_lm_head=args.include_lm_head,
    )