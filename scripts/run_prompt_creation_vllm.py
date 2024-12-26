import os
import torch
import re
from datasets import DatasetDict, load_dataset
from vllm import LLM, SamplingParams

SELECTED_KEYS_MEANING = {
    "rms": "speaking loudness",
    "speaking_rate": "speaking speed",
    "description": "metadata",
}

SYSTEM_PROMPT = """
You are tasked with generating a concise and accurate description of a speech sample based on provided speaking attributes. 

+ Use only the information provided in the attributes to construct the description.
+ Ensure that all provided attributes are included in the description.
+ The description should be grammatically correct, easy to understand, and concise.
+ Do not infer or include any additional details that are not explicitly stated in the attributes.

Format your response as follows:

<description>
{description}
</description>

### Example:
Given the speaking attributes:
- speaking loudness: speaking quietly
- speaking speed: moderately
- metadata: 30대 여자 목소리

### Output:
<description>
A soft-spoken woman in her 30s with a moderate speaking pace.
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
        add_generation_prompt=True,
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

    output_texts = [responses[idx].outputs[0].text for idx in range(len(responses))]
    output_tokens = [responses[idx].outputs[0].token_ids for idx in range(len(responses))]

    # only allow alphabets, numbers, spaces, and some special characters
    output_texts_refined = [
        ' '.join(re.findall(r"[a-zA-Z0-9\s.,'\"-]+", text.replace('<description>\n', '').replace('\n</description>', '').strip()))
        for text in output_texts
    ]
    
    batch["description"] = output_texts_refined
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

    if args.hf_dataset_name_output:
        dataset.push_to_hub(args.hf_dataset_name_output, private=True)
        print(f"Pushed to hub: {args.hf_dataset_name_output}")
    else:
        print("Not pushing to hub")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", type=str, required=True)
    parser.add_argument("--hf_dataset_name_output", type=str, required=False)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    run(args)
