from llm import LLM
from sampling_params import SamplingParams
import os
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/gz-data/model/Qwen3-4B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.7, max_token=256)

    base_prompts = [
        "Introduce yourself and summarize the main capabilities of a lightweight LLM inference engine.",
        "List all prime numbers within 100 and briefly explain how to verify primality.",
        "Write a concise explanation of tensor parallelism and when it helps inference throughput.",
        "Explain the difference between prefill and decode stages in autoregressive generation.",
        "Give a short overview of KV cache and why it improves decoding efficiency.",
        "Summarize how RoPE works in transformer attention in simple terms.",
        "Describe the tradeoff between latency and throughput for batched LLM serving.",
        "Explain why GPU utilization can stay low even when VRAM usage is high during inference.",
    ]
    prompts = [f"[sample {i:02d}] {prompt}" for i in range(8) for prompt in base_prompts]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize = False,
            add_generate_prompt = True
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    main()
