# --- MemoryManager class (injected) ---
from dataclasses import dataclass
from typing import Any
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.utils import trim_messages

SUMMARY_PREFIX = "[SUMMARY]"

# Token-based limits for the rolling SystemMessage summary
MAX_SUMMARY_TOKENS = 500      # hard ceiling for the summary message (system)
TARGET_SUMMARY_TOKENS = 350   # ask the model to aim for this size
TARGET_BULLETS = 10           # keep the structure readable

# Keep a small window of the most recent verbatim turns (before trimming)
RECENT_TURNS_TO_KEEP = 12

logger = logging.getLogger(__name__)

def messages_stats(messages):
    total = len(messages)
    chars = 0
    tool_msgs = 0
    ai_with_tools = 0
    for m in messages:
        content = getattr(m, "content", "")
        try:
            chars += len(content) if isinstance(content, str) else 0
        except Exception:
            pass
        if isinstance(m, ToolMessage):
            tool_msgs += 1
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            ai_with_tools += 1
    return {"count": total, "chars": chars, "tool_msgs": tool_msgs, "ai_with_tools": ai_with_tools}

@dataclass
class MemoryManager:
    max_model_tokens: int
    max_stored_messages: int
    summary_chunk_size: int
    tool_output_char_limit: int
    llm: Any

    def _is_summary_message(self, m):
        return isinstance(m, SystemMessage) and isinstance(m.content, str) and m.content.startswith(SUMMARY_PREFIX)

    def slice_current_turn(self, msgs):
        last_human_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], HumanMessage):
                last_human_idx = i
                break
        return msgs[last_human_idx:] if last_human_idx is not None else msgs

    def _truncate_tool_outputs(self, messages):
        trimmed = []
        for m in messages:
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > self.tool_output_char_limit:
                logger.info("Clamp ToolMessage: name=%s id=%s original_len=%d -> %d", getattr(m, "name", None), getattr(m, "tool_call_id", None), len(m.content), self.tool_output_char_limit)
                trimmed.append(ToolMessage(
                    content=m.content[:self.tool_output_char_limit] + "\n…(truncated)",
                    name=getattr(m, "name", None),
                    tool_call_id=getattr(m, "tool_call_id", None),
                ))
            else:
                trimmed.append(m)
        return trimmed

    def _keep_last_tool_cycle(self, messages):
        """Keep only the last assistant-with-tools group (and its tool replies) after the last Human."""
        # Find last HumanMessage
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        seg_start = 0 if last_human_idx is None else last_human_idx + 1
        prefix = messages[:seg_start]
        segment = messages[seg_start:]
        if not segment:
            return messages

        # Find last assistant-with-tools
        last_ai_idx = None
        last_ids = []
        for j in range(len(segment) - 1, -1, -1):
            m = segment[j]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                ids = []
                try:
                    for tc in m.tool_calls:
                        tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                        if tc_id:
                            ids.append(tc_id)
                except Exception:
                    pass
                last_ai_idx = j
                last_ids = ids
                break

        if last_ai_idx is None:
            return messages

        need = set(last_ids)
        new_segment = []
        # Drop earlier assistant-with-tools and any ToolMessages
        for k in range(0, last_ai_idx):
            m = segment[k]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue
            if isinstance(m, ToolMessage):
                continue
            new_segment.append(m)

        # Keep chosen assistant
        new_segment.append(segment[last_ai_idx])

        # Collect matching tools after it
        chosen_tools = []
        chosen_ids = set()
        tail_other = []
        for k in range(last_ai_idx + 1, len(segment)):
            m = segment[k]
            if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in need:
                chosen_tools.append(m)
                chosen_ids.add(getattr(m, "tool_call_id", None))
            else:
                tail_other.append(m)

        new_segment.extend(chosen_tools)
        for m in tail_other:
            if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in chosen_ids:
                continue
            new_segment.append(m)

        return prefix + new_segment

    def _sanitize_for_summary(self, messages):
        sanitized = []
        for m in messages:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                sanitized.append(AIMessage(content=str(getattr(m, "content", ""))))
            else:
                sanitized.append(m)
        return sanitized

    def _ensure_summary_message(self, messages):
        if messages and self._is_summary_message(messages[0]):
            return messages[0], messages[1:]
        return None, messages

    def _num_tokens(self, text: str) -> int:
        """Best-effort token count for a string using the LLM's counter; fallback to a rough heuristic."""
        try:
            # Many LangChain LLMs expose get_num_tokens
            return int(self.llm.get_num_tokens(text))
        except Exception:
            # Heuristic: ~4 chars per token (English)
            return max(1, len(text) // 4)

    def _clip_to_tokens(self, text: str, max_tokens: int) -> str:
        """Hard-clip text so its token count is <= max_tokens, using a couple of proportional passes."""
        if self._num_tokens(text) <= max_tokens:
            return text
        # Proportional cut, then refine up to 3 passes
        tokens = self._num_tokens(text)
        # Keep a small safety margin
        ratio = (max_tokens / tokens) * 0.98
        new_len = max(1, int(len(text) * ratio))
        clipped = text[:new_len]
        # Refine passes if still over
        for _ in range(3):
            if self._num_tokens(clipped) <= max_tokens:
                break
            clipped = clipped[: max(1, int(len(clipped) * 0.9))]
        if not clipped.endswith("…"):
            clipped += "…"
        return clipped

    def _compress_summary_tokens(self, prior_summary_text: str, new_chunk_messages: list) -> str:
        """
        Re-compress the prior summary + new chunk into a fresh, token-bounded summary.
        Returns plain text (no [SUMMARY] header).
        """
        sanitized_chunk = self._sanitize_for_summary(new_chunk_messages)

        prompt = [
            SystemMessage(content=(
                "You maintain a rolling conversation summary used as a SystemMessage for an LLM.\n"
                f"Rewrite the summary to be as informative as possible while staying UNDER {TARGET_SUMMARY_TOKENS} tokens and around {TARGET_BULLETS} bullet points.\n"
                "Keep ONLY durable facts, decisions, open questions, user preferences, and canonical names/ids.\n"
                "Prefer specificity over fluff; remove duplicates and obsolete items. If there is available room, include brief concrete examples."
            )),
            SystemMessage(content="Existing summary:\n" + (prior_summary_text or "(none)")),
            SystemMessage(content="New conversation chunk follows; integrate only durable, salient points.")
        ] + sanitized_chunk

        try:
            compressed = self.llm.invoke(prompt).content or ""
        except Exception as e:
            compressed = f"(compression failed: {e})"

        # Final enforcement: hard-clip to MAX_SUMMARY_TOKENS
        tokens_before = self._num_tokens(compressed)
        if tokens_before > MAX_SUMMARY_TOKENS:
            clipped = self._clip_to_tokens(compressed, MAX_SUMMARY_TOKENS)
            logger.info("Token-clip summary: %d -> %d tokens (max=%d)", tokens_before, self._num_tokens(clipped), MAX_SUMMARY_TOKENS)
            compressed = clipped
        else:
            logger.info("Compressed summary tokens=%d (target<=%d)", tokens_before, TARGET_SUMMARY_TOKENS)

        return compressed

    def summarize_history(self, messages):
        """Roll up the earliest chunk of history into a single [SUMMARY] SystemMessage."""
        summary_msg, rest = self._ensure_summary_message(messages)
        non_summary = list(rest)
        logger.info("Non-summary count=%d (threshold=%d)", len(non_summary), self.max_stored_messages)
        if len(non_summary) <= self.max_stored_messages:
            return messages
        chunk = non_summary[:self.summary_chunk_size]
        remainder = non_summary[self.summary_chunk_size:]

        logger.info("Summarizing history: chunk=%d; remainder=%d", len(chunk), len(remainder))
        sanitized_chunk = self._sanitize_for_summary(chunk)

        prior_summary_text = summary_msg.content[len(SUMMARY_PREFIX):].lstrip() if summary_msg else ""
        compressed = self._compress_summary_tokens(prior_summary_text, sanitized_chunk)
        new_summary_msg = SystemMessage(content=f"{SUMMARY_PREFIX}\n{compressed}")
        logger.info(
            "New summary SystemMessage: tokens=%d (max=%d), chars=%d",
            self._num_tokens(new_summary_msg.content), MAX_SUMMARY_TOKENS, len(new_summary_msg.content)
        )
        logger.info("New summary SystemMessage content:\n%s", new_summary_msg.content)
        return [new_summary_msg] + remainder

    def _drop_orphan_tools_and_invalid_groups(self, messages):
        """Ensure OpenAI tool message pairing invariants within the given view."""
        cleaned = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                group_ids = []
                try:
                    for tc in m.tool_calls:
                        tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                        if tc_id:
                            group_ids.append(tc_id)
                except Exception:
                    pass
                need = set(group_ids)
                found = set()
                tools_for_group = []
                tail_other = []
                j = i + 1
                while j < n and not isinstance(messages[j], AIMessage):
                    tm = messages[j]
                    if isinstance(tm, ToolMessage) and getattr(tm, "tool_call_id", None) in need:
                        tools_for_group.append(tm)
                        found.add(getattr(tm, "tool_call_id", None))
                    else:
                        tail_other.append(tm)
                    j += 1
                if need and need.issubset(found):
                    cleaned.append(m)
                    cleaned.extend(tools_for_group)
                    for x in tail_other:
                        if not (isinstance(x, ToolMessage) and getattr(x, "tool_call_id", None) in found):
                            cleaned.append(x)
                else:
                    logger.info("OpenAI repair: dropping assistant-with-tools; missing=%s", list(need - found))
                    for x in tail_other:
                        if not (isinstance(x, ToolMessage) and getattr(x, "tool_call_id", None) in need):
                            cleaned.append(x)
                i = j
            else:
                if isinstance(m, ToolMessage):
                    tid = getattr(m, "tool_call_id", None)
                    seen_valid_id = False
                    for prev in reversed(cleaned):
                        if isinstance(prev, AIMessage) and getattr(prev, "tool_calls", None):
                            ids_prev = []
                            try:
                                for tc in prev.tool_calls:
                                    tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else (tc.get("id") if isinstance(tc, dict) else None)
                                    if tc_id:
                                        ids_prev.append(tc_id)
                            except Exception:
                                pass
                            if tid in ids_prev:
                                seen_valid_id = True
                                break
                        if isinstance(prev, HumanMessage):
                            break
                    if seen_valid_id:
                        cleaned.append(m)
                    else:
                        logger.info("OpenAI repair: dropping orphan ToolMessage id=%s", tid)
                else:
                    cleaned.append(m)
                i += 1
        return cleaned

    def _tail_recent_turns(self, messages, n: int):
        """Return the last n messages from the *non-summary* portion."""
        summary_msg, rest = self._ensure_summary_message(messages)
        tail = rest[-n:] if n > 0 else []
        return ([summary_msg] if summary_msg else []) + tail

    def prepare_for_model(self, messages):
        """Single-pass prep: clamp tool outputs, keep last tool cycle, trim by tokens, and repair pairs."""
        before = messages_stats(messages)
        # Clamp tool outputs
        msgs = self._truncate_tool_outputs(messages)
        # Keep only last tool cycle after last Human
        msgs = self._keep_last_tool_cycle(msgs)

        # Ensure we always keep a small recent window verbatim
        protected_tail = self._tail_recent_turns(msgs, RECENT_TURNS_TO_KEEP)
        # Build a set of ids to compare object identity safely
        protected_ids = {id(m) for m in protected_tail}
        # Move protected_tail to the end (stable order), keeping everything else before
        base = [m for m in msgs if id(m) not in protected_ids]
        msgs = base + protected_tail

        # Use trim_messages with strategy='last'
        try:
            trimmed = trim_messages(
                messages=msgs,
                max_tokens=self.max_model_tokens,
                strategy="last",
                allow_partial=False,
                token_counter=self.llm,
            )
        except Exception as e:
            trimmed = msgs[-self.max_stored_messages:]
            logger.warning("Trim fallback due to %s", e)

        # Repair for OpenAI tool message invariants
        cleaned = self._drop_orphan_tools_and_invalid_groups(trimmed)
        after = messages_stats(cleaned)

        # Observability: if we start with a [SUMMARY], log its token count
        summary_msg, _ = self._ensure_summary_message(cleaned)
        if summary_msg:
            try:
                logger.info("Summary tokens now=%d", self._num_tokens(summary_msg.content))
            except Exception:
                pass

        logger.info("prepare_for_model: before=%s -> after=%s (token_budget=%d)", before, after, self.max_model_tokens)
        return cleaned
