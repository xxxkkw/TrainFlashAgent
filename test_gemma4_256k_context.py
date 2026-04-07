from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


DEFAULT_MODEL_PATH = "/data/xxxkw/gemma-4-31B-it"
DEFAULT_TARGET_INPUT_TOKENS = 256000
DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_NEEDLE = "GEMMA4-256K-NEEDLE-31415926"
DEFAULT_NEEDLE_POSITION = 0.72

SYSTEM_PROMPT = "你是一个严格的长上下文检索助手。你只能输出文档里明确出现过的最终答案。"
INTRO = (
    "下面是一份超长文档，真正需要回答的密钥只会在文档中出现一次。\n"
    "请完整阅读后回答最后的问题。\n"
    "回答规则：\n"
    "1. 只输出密钥原文。\n"
    "2. 不要解释，不要复述题目，不要添加任何额外字符。\n"
    "3. 如果没有找到密钥，输出 NOT_FOUND。\n\n"
    "文档开始：\n"
)
NEEDLE_TEMPLATE = "\n[关键信息] 本次 256K 长上下文测试的唯一密钥是：{needle}\n"
QUESTION = "\n文档结束。\n\n问题：上面长文档中唯一出现的密钥是什么？只输出密钥原文。"
FILLER_UNIT = (
    "这是一段用于 256K 上下文压力测试的填充文本。"
    "它不包含答案，请继续向后阅读并忽略这里的内容。"
    "真正需要记住的关键信息只会出现一次。\n"
)
TAIL_UNIT = "补充段落。\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backend",
        choices=["build-only", "transformers", "openai", "vllm-token-ids"],
        default="build-only",
    )
    parser.add_argument("--target-input-tokens", type=int, default=DEFAULT_TARGET_INPUT_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--needle-position", type=float, default=DEFAULT_NEEDLE_POSITION)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output-file", default="")
    return parser.parse_args()


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def build_user_content(filler_repeats: int, tail_repeats: int, needle: str, needle_position: float) -> str:
    prefix_fillers = int(filler_repeats * needle_position)
    suffix_fillers = filler_repeats - prefix_fillers
    prefix = FILLER_UNIT * prefix_fillers
    suffix = FILLER_UNIT * suffix_fillers
    tail = TAIL_UNIT * tail_repeats
    return "".join(
        [
            INTRO,
            prefix,
            NEEDLE_TEMPLATE.format(needle=needle),
            suffix,
            tail,
            QUESTION,
        ]
    )


def build_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def count_prompt_tokens(tokenizer, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return int(encoded["input_ids"].shape[-1])


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return encoded["input_ids"][0].tolist()


def fit_prompt_to_target(tokenizer, needle: str, target_input_tokens: int, needle_position: float):
    cache: dict[tuple[int, int], tuple[int, list[dict[str, str]], str]] = {}

    def measure(filler_repeats: int, tail_repeats: int):
        key = (filler_repeats, tail_repeats)
        if key in cache:
            return cache[key]
        user_content = build_user_content(
            filler_repeats=filler_repeats,
            tail_repeats=tail_repeats,
            needle=needle,
            needle_position=needle_position,
        )
        messages = build_messages(user_content)
        token_count = count_prompt_tokens(tokenizer, messages)
        cache[key] = (token_count, messages, user_content)
        return cache[key]

    filler_low = 0
    filler_high = 1
    filler_tokens, _, _ = measure(filler_high, 0)
    while filler_tokens < target_input_tokens:
        filler_low = filler_high
        filler_high *= 2
        filler_tokens, _, _ = measure(filler_high, 0)

    best_filler = 0
    best_tokens = 0
    best_messages: list[dict[str, str]] = []
    best_user_content = ""

    left = filler_low
    right = filler_high
    while left <= right:
        mid = (left + right) // 2
        token_count, messages, user_content = measure(mid, 0)
        if token_count <= target_input_tokens:
            best_filler = mid
            best_tokens = token_count
            best_messages = messages
            best_user_content = user_content
            left = mid + 1
        else:
            right = mid - 1

    tail_low = 0
    tail_high = 1
    tail_tokens, _, _ = measure(best_filler, tail_high)
    while tail_tokens < target_input_tokens:
        tail_low = tail_high
        tail_high *= 2
        tail_tokens, _, _ = measure(best_filler, tail_high)

    left = tail_low
    right = tail_high
    while left <= right:
        mid = (left + right) // 2
        token_count, messages, user_content = measure(best_filler, mid)
        if token_count <= target_input_tokens:
            best_tokens = token_count
            best_messages = messages
            best_user_content = user_content
            left = mid + 1
        else:
            right = mid - 1

    prompt = render_prompt(tokenizer, best_messages)
    return {
        "messages": best_messages,
        "user_content": best_user_content,
        "prompt": prompt,
        "prompt_tokens": best_tokens,
        "prompt_chars": len(prompt),
        "remaining_context": max(0, target_input_tokens - best_tokens),
    }


def maybe_write_prompt(output_file: str, prompt: str) -> None:
    if not output_file:
        return
    output_path = Path(output_file).expanduser().resolve()
    output_path.write_text(prompt, encoding="utf-8")


def resolve_served_model_name(client, requested_name: str) -> str:
    if requested_name:
        return requested_name
    models = client.models.list().data
    if not models:
        raise RuntimeError("未从服务端发现可用模型，请用 --served-model-name 显式指定模型名。")
    return models[0].id


def run_openai_backend(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    model_name = args.served_model_name or args.model_path
    if args.served_model_name:
        print(f"stage=using_explicit_model_name model={model_name}", flush=True)
    else:
        print(f"stage=using_model_path_as_model_name model={model_name}", flush=True)
    print("stage=openai_request_send_begin", flush=True)

    if args.stream:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=0,
            stream=True,
        )
        chunks: list[str] = []
        first_chunk_seen = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and not first_chunk_seen:
                print("stage=openai_first_chunk_received", flush=True)
                print("response_begin", flush=True)
                first_chunk_seen = True
            if delta:
                chunks.append(delta)
                print(delta, end="", flush=True)
        if first_chunk_seen:
            print(file=sys.stdout, flush=True)
            print("response_end", flush=True)
        else:
            print("stage=openai_stream_finished_without_text", flush=True)
        return "".join(chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=args.max_new_tokens,
        temperature=0,
    )
    print("stage=openai_response_received", flush=True)
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def run_vllm_token_ids_backend(args: argparse.Namespace, tokenizer, messages: list[dict[str, str]]) -> str:
    model_name = args.served_model_name or args.model_path
    prompt_token_ids = build_prompt_token_ids(tokenizer, messages)
    payload = {
        "model": model_name,
        "prompt": prompt_token_ids,
        "max_tokens": args.max_new_tokens,
        "temperature": 0,
        "stream": args.stream,
    }
    request = urllib.request.Request(
        url=f"{args.base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    print(f"stage=using_pretokenized_prompt token_count={len(prompt_token_ids)}", flush=True)
    print("stage=vllm_completions_request_send_begin", flush=True)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if args.stream:
            chunks: list[str] = []
            first_chunk_seen = False
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                text = event["choices"][0].get("text", "")
                if text and not first_chunk_seen:
                    print("stage=vllm_completions_first_chunk_received", flush=True)
                    print("response_begin", flush=True)
                    first_chunk_seen = True
                if text:
                    chunks.append(text)
                    print(text, end="", flush=True)
            if first_chunk_seen:
                print(file=sys.stdout, flush=True)
                print("response_end", flush=True)
            else:
                print("stage=vllm_completions_stream_finished_without_text", flush=True)
            return "".join(chunks)
        body = json.loads(response.read().decode("utf-8"))
    print("stage=vllm_completions_response_received", flush=True)
    return body["choices"][0]["text"]


def run_transformers_backend(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    if not (0 < args.needle_position < 1):
        raise ValueError("--needle-position 必须在 0 和 1 之间。")
    tokenizer = load_tokenizer(args.model_path)
    result = fit_prompt_to_target(
        tokenizer=tokenizer,
        needle=args.needle,
        target_input_tokens=args.target_input_tokens,
        needle_position=args.needle_position,
    )
    if result["prompt_tokens"] + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            "输入长度加输出长度超过 max_model_len，请降低 --target-input-tokens 或 --max-new-tokens。"
        )
    maybe_write_prompt(args.output_file, result["prompt"])

    print(f"model_path={args.model_path}")
    print(f"backend={args.backend}")
    print(f"prompt_tokens={result['prompt_tokens']}")
    print(f"target_input_tokens={args.target_input_tokens}")
    print(f"remaining_to_target={result['remaining_context']}")
    print(f"max_model_len={args.max_model_len}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"prompt_chars={result['prompt_chars']}")
    print(f"needle={args.needle}")
    print(f"timeout={args.timeout}")
    print(f"stream={args.stream}")

    if args.backend == "build-only":
        print("status=prompt_built_only")
        return

    if args.backend == "openai":
        answer = run_openai_backend(args, result["messages"])
    elif args.backend == "vllm-token-ids":
        answer = run_vllm_token_ids_backend(args, tokenizer, result["messages"])
    else:
        answer = run_transformers_backend(args, result["messages"])

    normalized_answer = answer.strip()
    if not args.stream:
        print("response_begin")
        print(normalized_answer)
        print("response_end")
    print(f"needle_found={args.needle in normalized_answer}")


if __name__ == "__main__":
    main()
