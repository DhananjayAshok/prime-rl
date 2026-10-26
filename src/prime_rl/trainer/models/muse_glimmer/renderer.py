# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
"""A `renderers` renderer for meta-models/Muse-Glimmer-30B, loaded as a custom renderer:

    [orchestrator.renderer]
    name = "custom"
    import_path = "prime_rl.trainer.models.muse_glimmer.renderer.MuseGlimmerRenderer"

The `renderers` package ships no Muse Glimmer renderer, and its `DefaultRenderer` would stop only
at `<|end_of_text|>` (not the turn close `<|eot|>`) and return the reasoning channel as content.

Mirrors the checkpoint's `chat_template.jinja` (the "ATEM" template): every message is
`<|start|>{header}<|message|>{body}{close}`; an assistant turn is one or more channels
`<|start|>assistant to={recipient}<|message|>...`, where `to=self` is reasoning (closed by
`<|eom|>`), `to=user` the answer (closed by `<|eot|>`, which ends the turn) and `to=<tool>` an ATEM
tool call. The generation prompt is `<|start|>assistant`; the model writes the rest, including
the `<|start|>assistant to=user<|message|>` header after its reasoning. Parsing follows vLLM's
`muse_glimmer` reasoning parser (reasoning = the `to=self` bodies, content = the `to=user` body),
but on token ids.

The default system block's "Current date" (the template uses `strftime_now`) comes from
`current_date`, or today's date when unset, as in the template.
"""

import json
import re
from datetime import date
from typing import Any, Literal

from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    Tokenizer,
    ToolCallParseStatus,
    ToolSpec,
    _content_mask_or_empty,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import BaseRendererConfig

DEFAULT_SYSTEM = "You are a helpful AI assistant."
TOOL_INTRO = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the following:\n'
    '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n'
    "</atem:function_calls>\n\n"
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. "
    "Note that spaces for string values are not stripped. The output is not expected to be valid XML and is "
    "parsed with regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)
TOOL_EXAMPLE = (
    "\n\nHere's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather "
    "than `example_tool_name.example_function_name`)\n\n"
    "to=example_tool_name.example_function_name\n\n"
    '<atem:function_calls>\n<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second parameter\n'
    'that can span\n"multiple" lines\n</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)
RECIPIENT_RE = re.compile(r"to=([^\s<]+)")
INVOKE_RE = re.compile(r'<atem:invoke name="([^"]*)">(.*?)</atem:invoke>', re.DOTALL)
PARAMETER_RE = re.compile(r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>', re.DOTALL)


class MuseGlimmerRendererConfig(BaseRendererConfig):
    name: Literal["muse-glimmer"] = "muse-glimmer"
    _template_fields = frozenset({"reasoning_strength", "knowledge_cutoff", "current_date"})

    reasoning_strength: str = "high"
    """The `Reasoning strength: ...` line of the system block (the template's `reasoning_strength`
    kwarg; the template defaults to "high")."""

    knowledge_cutoff: str = "2026-01-04"
    """Only used in the default system block (no system message)."""

    current_date: str | None = None
    """Only used in the default system block; None means today's date (the template's
    `strftime_now('%Y-%m-%d')`)."""


def tojson(*, value: Any) -> str:
    """The chat template's `tojson` (transformers: json.dumps with ensure_ascii=False)."""
    return json.dumps(value, ensure_ascii=False)


def content_text(*, content: Any) -> str:
    """The template's `render_content`, text parts only."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif part.get("type") in ("image", "image_url", "video"):
            raise ValueError("MuseGlimmerRenderer renders text only")
    return "".join(parts)


def function_of(*, tool: dict) -> dict:
    return tool["function"] if isinstance(tool.get("function"), dict) else tool


def tool_namespaces(*, tools: list[ToolSpec] | None) -> list[str]:
    namespaces = []
    for tool in tools or []:
        namespace = function_of(tool=tool)["name"].split(".")[0]
        if namespace not in namespaces:
            namespaces.append(namespace)
    return namespaces


def tool_defs_text(*, tools: list[ToolSpec]) -> str:
    """The template's `render_tool_defs`."""
    text = TOOL_INTRO
    for namespace in tool_namespaces(tools=tools):
        text += '{"name": ' + tojson(value=namespace) + ', "description": ' + tojson(value="") + "}\n"
    text += "// Function schemas"
    for tool in tools:
        fn = function_of(tool=tool)
        text += (
            '\n{"name": '
            + tojson(value=fn["name"])
            + ', "description": '
            + tojson(value=fn.get("description", ""))
            + ', "parameters": '
            + tojson(value=fn.get("parameters", {}))
            + "}"
        )
    return text + TOOL_EXAMPLE


def recipients_text(*, tools: list[ToolSpec] | None) -> str:
    """The template's `render_system_meta`."""
    recipients = ['"self"'] + [f'"{namespace}.*"' for namespace in tool_namespaces(tools=tools)] + ['"user"']
    return "# Valid recipients: " + ", ".join(recipients) + "."


def atem_value(*, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return tojson(value=value)
    return str(value)


def atem_call_text(*, tool_call: dict) -> tuple[str, str]:
    """(function name, the template's `render_atem` body) for one tool call."""
    fn = tool_call.get("function") or tool_call
    arguments = fn.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Muse Glimmer tool-call arguments must be a mapping")
    text = '<atem:function_calls>\n<atem:invoke name="' + fn["name"] + '">\n'
    for key, value in arguments.items():
        text += '<atem:parameter name="' + key + '">' + atem_value(value=value) + "</atem:parameter>\n"
    return fn["name"], text + "</atem:invoke>\n</atem:function_calls>"


def tool_message_name(*, messages: list[Message], index: int) -> str:
    """The template's tool-name lookup: the message's `name`, else the name of the call whose id
    is its `tool_call_id`, else the id itself."""
    message = messages[index]
    if message.get("name"):
        return message["name"]
    call_id = message.get("tool_call_id")
    name = call_id or ""
    for other in messages:
        for tool_call in other.get("tool_calls") or []:
            if call_id is not None and tool_call.get("id") == call_id:
                name = (tool_call.get("function") or tool_call)["name"]
    return name


class MuseGlimmerRenderer:
    config_class = MuseGlimmerRendererConfig

    def __init__(self, tokenizer: Tokenizer, config: MuseGlimmerRendererConfig | None = None):
        self._tokenizer = tokenizer
        self.config = config or MuseGlimmerRendererConfig()
        # The template keeps every past reasoning channel, so extending a rollout is always safe.
        self.effective_thinking_retention = resolve_thinking_retention(self.config, "all")
        self._bos = self._token_id(token="<|begin_of_text|>")
        self._start = self._token_id(token="<|start|>")
        self._message = self._token_id(token="<|message|>")
        self._eom = self._token_id(token="<|eom|>")
        self._eot = self._token_id(token="<|eot|>")
        self._end_of_text = self._token_id(token="<|end_of_text|>")

    def _token_id(self, *, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id == getattr(self._tokenizer, "unk_token_id", None):
            raise ValueError(f"special token {token!r} is not in the tokenizer's vocabulary")
        return token_id

    # ------------------------------------------------------------------ render

    def _system_body(self, *, text: str | None, tools: list[ToolSpec] | None) -> list[tuple[str, bool]]:
        """(text, is_content) segments of a system block's body. `text` None: the default block."""
        if text is None:
            today = self.config.current_date or date.today().strftime("%Y-%m-%d")
            preamble = f"{DEFAULT_SYSTEM}\nKnowledge cutoff: {self.config.knowledge_cutoff}.\nCurrent date: {today}."
            segments = [(preamble, False), ("\n\nReasoning strength: " + self.config.reasoning_strength + ".", False)]
        else:
            for old, new in (
                ("Reasoning effort", "Reasoning strength"),
                ("Reasoning Effort", "Reasoning Strength"),
                ("reasoning effort", "reasoning strength"),
                ("REASONING EFFORT", "REASONING STRENGTH"),
            ):
                text = text.replace(old, new)
            segments = [(text, True)]
            if "reasoning strength" not in text.lower():
                segments.append(("\n\nReasoning strength: " + self.config.reasoning_strength + ".", False))
        if tools:
            segments.append(("\n\n" + tool_defs_text(tools=tools), False))
        segments.append(("\n\n" + recipients_text(tools=tools), False))
        return segments

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content: list[bool] = []

        def emit_special(token_id: int, msg_idx: int, is_sampled: bool) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content.append(is_sampled)

        def emit_run(segments: list[tuple[str, bool, bool]], msg_idx: int) -> None:
            """One text run between special tokens, BPE'd in one pass like the template's output;
            segments are (text, is_sampled, is_content)."""
            segments = [s for s in segments if s[0]]
            if not segments:
                return
            with_sampled = attribute_text_segments(self._tokenizer, [(text, s) for text, s, _ in segments])
            with_content = attribute_text_segments(self._tokenizer, [(text, c) for text, _, c in segments])
            for (token_id, is_sampled), (_, is_content) in zip(with_sampled, with_content, strict=True):
                tokens.append(token_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content.append(is_content)

        def emit_block(header: str, body: list[tuple[str, bool]], close: int, msg_idx: int) -> None:
            """A non-assistant block: nothing in it is sampled."""
            emit_special(self._start, msg_idx, False)
            emit_run([(header, False, False)], msg_idx)
            emit_special(self._message, msg_idx, False)
            emit_run([(text, False, is_content) for text, is_content in body], msg_idx)
            emit_special(close, msg_idx, False)

        def emit_channel(recipient: str, body: str, close: int, msg_idx: int, opens_turn: bool) -> None:
            """An assistant channel. The turn's first `<|start|>assistant` is the generation
            prompt; everything after it (later channel headers included) is sampled."""
            emit_special(self._start, msg_idx, not opens_turn)
            emit_run([("assistant", not opens_turn, not opens_turn), (" to=" + recipient, True, True)], msg_idx)
            emit_special(self._message, msg_idx, True)
            emit_run([(body, True, True)], msg_idx)
            emit_special(close, msg_idx, True)

        emit_special(self._bos, -1, False)
        if not any(m.get("role") == "system" for m in messages):
            emit_block("system", self._system_body(text=None, tools=tools), self._eot, -1)

        for i, message in enumerate(messages):
            role = message.get("role")
            next_same_role = i + 1 < len(messages) and messages[i + 1].get("role") == role
            end_token = self._eom if next_same_role else self._eot
            if role == "system":
                body = self._system_body(text=content_text(content=message.get("content")), tools=tools)
                emit_block("system", body, self._eot, i)
            elif role == "user":
                emit_block("user", [(content_text(content=message.get("content")), True)], self._eot, i)
            elif role == "tool":
                name = tool_message_name(messages=messages, index=i)
                body = [
                    (f'<tool_output name="{name}">\n', False),
                    (content_text(content=message.get("content")), True),
                    ("\n</tool_output>", False),
                ]
                emit_block("tool " + name, body, self._eot, i)
            elif role == "assistant":
                opens_turn = True
                reasoning = message.get("reasoning_content")
                if reasoning:
                    emit_channel("self", reasoning, self._eom, i, opens_turn)
                    opens_turn = False
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    for j, tool_call in enumerate(tool_calls):
                        name, body = atem_call_text(tool_call=tool_call)
                        close = end_token if j == len(tool_calls) - 1 else self._eom
                        emit_channel(name, body, close, i, opens_turn)
                        opens_turn = False
                else:
                    recipient = message.get("recipient") or "user"
                    end_turn = message.get("end_turn")
                    if end_turn is None:
                        end_turn = recipient == "user"
                    body = content_text(content=message.get("content"))
                    emit_channel(recipient, body, self._eot if end_turn else self._eom, i, opens_turn)
            else:
                raise ValueError(f"MuseGlimmerRenderer cannot render role {role!r}")

        if add_generation_prompt:
            emit_special(self._start, -1, False)
            emit_run([("assistant", False, False)], -1)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=_content_mask_or_empty(self._tokenizer, content),
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt).token_ids

    def get_stop_token_ids(self) -> list[int]:
        # <|eom|> closes a channel inside a turn (reasoning, a non-final tool call): not a stop.
        return [self._eot, self._end_of_text]

    # ------------------------------------------------------------------ parse

    def _open_channel_in_prompt(self, *, prompt_ids: list[int] | None) -> tuple[list[int], str | None]:
        """What the prompt leaves open at its end: (header tokens after the last `<|start|>`, None)
        when a channel header is still being written, or ([], recipient) when the prompt prefilled
        a channel's `<|message|>` (e.g. `...to=self<|message|>`)."""
        if not prompt_ids or self._start not in prompt_ids:
            return [], None
        last_start = len(prompt_ids) - 1 - prompt_ids[::-1].index(self._start)
        tail = prompt_ids[last_start + 1 :]
        if self._eom in tail or self._eot in tail:
            return [], None
        if self._message in tail:
            header = tail[: tail.index(self._message)]
            return [], self._recipient(header=header)
        return tail, None

    def _recipient(self, *, header: list[int]) -> str:
        match = RECIPIENT_RE.search(self._tokenizer.decode(header, skip_special_tokens=False))
        return match.group(1) if match else "user"

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,
    ) -> ParsedResponse:
        stop_ids = {self._eot, self._end_of_text}
        ids = list(token_ids)
        stopped = False
        for k, token_id in enumerate(ids):
            if token_id in stop_ids:
                ids, stopped = ids[:k], True
                break

        header, recipient = self._open_channel_in_prompt(prompt_ids=prompt_ids)
        channels: list[tuple[str, list[int], bool]] = []  # (recipient, body ids, closed)
        body: list[int] = []
        stray: list[int] = []  # tokens outside any channel (no header was written)
        for token_id in ids:
            if recipient is None:
                if token_id == self._start:
                    stray.extend(header)
                    header = []
                elif token_id == self._message:
                    recipient, body = self._recipient(header=header), []
                    header = []
                else:
                    header.append(token_id)
            elif token_id in (self._eom, self._start):
                # A new header without <|eom|> is a model defect vLLM's parser also tolerates.
                channels.append((recipient, body, token_id == self._eom))
                recipient, body, header = None, [], []
            else:
                body.append(token_id)
        if recipient is not None:
            channels.append((recipient, body, stopped))
        stray.extend(header)

        decode = lambda part: self._tokenizer.decode(part, skip_special_tokens=False)  # noqa: E731
        reasoning = [decode(b) for r, b, _ in channels if r == "self"]
        answer = [decode(b) for r, b, _ in channels if r == "user"]
        if not channels:
            answer = [self._tokenizer.decode(stray, skip_special_tokens=True)]
        tool_calls = [
            self._parse_tool_call(recipient=r, text=decode(b), closed=closed, tools=tools)
            for r, b, closed in channels
            if r not in ("self", "user")
        ]
        reasoning_open = bool(channels) and channels[-1][0] == "self" and not channels[-1][2]
        return ParsedResponse(
            content="".join(answer),
            reasoning_content="".join(reasoning) if reasoning else None,
            tool_calls=tool_calls,
            reasoning_complete=not reasoning_open,
        )

    def _parse_tool_call(
        self, *, recipient: str, text: str, closed: bool, tools: list[ToolSpec] | None
    ) -> ParsedToolCall:
        invoke = INVOKE_RE.search(text)
        if invoke is None:
            status = ToolCallParseStatus.MALFORMED_STRUCTURE if closed else ToolCallParseStatus.UNCLOSED_BLOCK
            return ParsedToolCall(raw=text, name=recipient, status=status)
        name = invoke.group(1) or recipient
        string_params = set()
        for tool in tools or []:
            fn = function_of(tool=tool)
            if fn["name"] == name:
                properties = (fn.get("parameters") or {}).get("properties") or {}
                string_params = {key for key, schema in properties.items() if schema.get("type") == "string"}
        arguments = {}
        for key, value in PARAMETER_RE.findall(invoke.group(2)):
            if key in string_params:
                arguments[key] = value
                continue
            try:
                arguments[key] = json.loads(value)
            except ValueError:
                arguments[key] = value
        status = ToolCallParseStatus.OK if name else ToolCallParseStatus.MISSING_NAME
        return ParsedToolCall(raw=text, name=name, arguments=arguments, status=status)

    # ------------------------------------------------------------------ bridge

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        if not previous_prompt_ids or not new_messages or reject_assistant_in_extension(new_messages):
            return None
        if any(m.get("role") not in ("user", "tool", "system") for m in new_messages):
            return None
        if should_rerender_for_thinking_retention(self.effective_thinking_retention, new_messages):
            return None
        # A turn cut inside its reasoning channel has no canonical close to synthesize.
        if not self.parse_response(previous_completion_ids, prompt_ids=previous_prompt_ids).reasoning_complete:
            return None
        previous_ids = trim_to_turn_close(
            previous_prompt_ids, previous_completion_ids, {self._eot, self._end_of_text}, synthesize_close=self._eot
        )
        if previous_ids is None:
            return None
        # Render the new messages as a continuation (no BOS / default system block): render them
        # after a stand-in system message and keep what follows it.
        anchor: Message = {"role": "system", "content": ""}
        full = self.render([anchor, *new_messages], tools=tools, add_generation_prompt=True)
        start = full.message_indices.index(1)
        ext_indices = [idx - 1 if idx > 0 else -1 for idx in full.message_indices[start:]]
        ext_content = full.is_content[start:] if full.is_content else []
        total = len(previous_ids) + len(full.token_ids) - start
        return RenderedTokens(
            token_ids=previous_ids + full.token_ids[start:],
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total,
            is_content=[False] * len(previous_ids) + ext_content if ext_content else [],
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )


__all__ = ["MuseGlimmerRenderer", "MuseGlimmerRendererConfig"]
