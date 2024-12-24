import os
import torch
from datasets import DatasetDict, load_dataset
from vllm import LLM, SamplingParams

SELECTED_KEYS_MEANING = {
    "rms": "speaking loudness",
    "speaking_rate": "speaking speed",
    "description": "metadata",
}

SYSTEM_PROMPT = """
Given a speaking attributes of a speech, create a description of the speech. 
You should answer in the following format:

<description>
{description}
</description>
"""

def create_description(batch, args, llm):
    descriptions = []
    for idx in range(len(batch["rms"])):
        description = []
        for key in SELECTED_KEYS_MEANING:
            if key in batch:
                value = batch[key][idx]
                meaning = SELECTED_KEYS_MEANING[key]
                # Directly use the value from the batch
                description.append(f"{meaning}: {value}")
        descriptions.append("### Speaking Attributes\n" + "\n".join(description))
    
    convs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": descriptions[idx]},
        ]
        for idx in range(len(descriptions))
    ]
    tokenizer = llm.get_tokenizer()
    templated_convs = tokenizer.apply_chat_template(
        convs,
        tokenize=False,
    )

    completions = [[] for _ in range(len(descriptions))]
    completion_tokens = [[] for _ in range(len(descriptions))]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    print(convs[0])
    print(responses[0])
    print("===============")
    # batch["description"] = descriptions
    return batch

def run(args):
    # Load dataset
    dataset = load_dataset(args.hf_dataset_name)
    print("Loaded dataset")

    # LLM load
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print("Loaded LLM")

    # Map dataset
    dataset = dataset.map(
        create_description,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs={"args": args, "llm": llm},
        desc="Creating descriptions",
        load_from_cache_file=False,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="beomi/EXAONE-3.5-7.8B-Instruct-Llamafied")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    run(args)
