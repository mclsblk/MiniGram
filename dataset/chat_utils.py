import json
from typing import Any, Iterable, Optional


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False).strip()
    return str(content).strip()


def _parse_json_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def extract_tool_call_text(turn: dict) -> str:
    tool_call_texts = []
    for key in ("tool_call", "tool_calls"):
        value = stringify_content(turn.get(key))
        if value:
            tool_call_texts.append(value)
    return "\n".join(tool_call_texts)


def _extract_tools(conversations: Iterable[dict]) -> Optional[Any]:
    for turn in conversations:
        if not isinstance(turn, dict) or turn.get("role") != "system":
            continue
        for key in ("tools", "functions"):
            tools = _parse_json_if_possible(turn.get(key))
            if tools:
                return tools
        return None
    return None


def normalize_conversations(sample_or_conversations, include_empty_assistant: bool = False):
    conversations = (
        sample_or_conversations.get("conversations")
        if isinstance(sample_or_conversations, dict)
        else sample_or_conversations
    )
    if conversations is None:
        raise ValueError("Expected a sample with conversations or a conversations list.")

    tools = _extract_tools(conversations)
    messages = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        role = turn.get("role")
        content = stringify_content(turn.get("content"))
        tool_call_text = extract_tool_call_text(turn)

        if role == "assistant":
            assistant_parts = [part for part in (content, tool_call_text) if part]
            content = "\n".join(assistant_parts).strip()
            if not content and not include_empty_assistant:
                continue
        elif role != "assistant" and not content:
            continue

        if role:
            messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("Conversation produced no usable chat messages.")
    return messages, tools


def inject_system_prompt(messages, system_prompt: Optional[str] = None):
    system_content = stringify_content(system_prompt)
    if not system_content:
        return messages

    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        existing_content = stringify_content(messages[0].get("content"))
        if existing_content == system_content:
            return messages

        merged_messages = list(messages)
        merged_messages[0] = {
            "role": "system",
            "content": "\n".join(part for part in (system_content, existing_content) if part),
        }
        return merged_messages

    return [{"role": "system", "content": system_content}] + messages


def render_chat_prompt(tokenizer, messages, tools=None, add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs)

    prompt = ""
    for message in messages:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    return prompt


def build_sft_prompt(tokenizer, sample_or_conversations) -> str:
    messages, tools = normalize_conversations(sample_or_conversations)
    return render_chat_prompt(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        add_generation_prompt=False,
    )


def build_generation_prompt(
    tokenizer,
    user_text: Optional[str] = None,
    history: Optional[list] = None,
    with_history: bool = False,
    sample_or_conversations=None,
) -> str:
    if sample_or_conversations is not None:
        messages, tools = normalize_conversations(sample_or_conversations)
    else:
        messages = []
        if with_history and history:
            messages.extend(history)
        messages.append({"role": "user", "content": stringify_content(user_text)})
        tools = None

    return render_chat_prompt(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
    )
