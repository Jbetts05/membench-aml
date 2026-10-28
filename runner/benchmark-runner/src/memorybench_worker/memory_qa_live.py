from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MemorySearchOptions, MemoryStoreDefaultDefinition, MemoryStoreDefaultOptions
from azure.identity import DefaultAzureCredential
from state_bench.client import LLMClient

PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
ANSWER_DEPLOYMENT = os.environ.get("ANSWER_DEPLOYMENT", "")
JUDGE_DEPLOYMENT = os.environ.get("JUDGE_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.environ.get("EMBEDDING_DEPLOYMENT", "")
STORE = os.environ.get("MEMORY_STORE_NAME", "memorybench_live_qa")
if not ANSWER_DEPLOYMENT:
    raise RuntimeError("Set ANSWER_DEPLOYMENT before running MemoryBench QA.")
if not JUDGE_DEPLOYMENT:
    raise RuntimeError("Set JUDGE_DEPLOYMENT before running MemoryBench QA.")
if not EMBEDDING_DEPLOYMENT:
    raise RuntimeError("Set EMBEDDING_DEPLOYMENT before running MemoryBench QA.")
ANSWER_PROMPT_VERSION = "longmemeval-answer-v3-current-date"
JUDGE_PROMPT_VERSION = "longmemeval-official-evaluate-qa"
EVIDENCE_JUDGE_PROMPT_VERSION = "longmemeval-evidence-support-v1"
PRIMARY_MLFLOW_METRICS = {
    "foundry_memory_percent_correct",
    "full_transcript_percent_correct",
    "selected_transcript_percent_correct",
    "oracle_context_percent_correct",
    "question_only_percent_correct",
    "case_count",
    "answer_accuracy_lift_pp",
}


def benchmark_label(manifest: dict[str, Any]) -> str:
    return str(manifest.get("benchmark") or manifest.get("benchmarkId") or "memory-qa")


def artifact_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "memory-qa"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def truncate_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def exception_diagnostics(exc: Exception, *, message_limit: int = 1000, body_limit: int = 1000) -> dict[str, Any]:
    details: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": truncate_text(exc, message_limit),
    }

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code

    error = getattr(exc, "error", None)
    if error is not None:
        error_code = getattr(error, "code", None)
        error_message = getattr(error, "message", None)
        if error_code is not None:
            details["error_code"] = error_code
        if error_message is not None:
            details["error_message"] = truncate_text(error_message, message_limit)

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status is not None and "status_code" not in details:
            details["status_code"] = response_status
        reason = getattr(response, "reason", None)
        if reason is not None:
            details["reason"] = str(reason)
        headers = getattr(response, "headers", None) or {}
        for header in ("x-ms-error-code", "x-ms-request-id", "request-id", "retry-after"):
            value = headers.get(header) if hasattr(headers, "get") else None
            if value is not None:
                details[header.replace("-", "_")] = str(value)
        try:
            body = response.text()
        except Exception:
            body = None
        if body:
            details["response_body"] = truncate_text(body, body_limit)

    return details


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_cases(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "benchmark-set.json").read_text(encoding="utf-8"))
    cases = json.loads((root / "longmemeval-s-20-raw.json").read_text(encoding="utf-8"))
    return manifest, cases


def normalize_case(item: dict[str, Any], max_sessions: int = 9, history_mode: str = "full") -> dict[str, Any]:
    answer_ids = set(item.get("answer_session_ids") or [])
    total_sessions = len(item.get("haystack_sessions") or [])
    if history_mode == "full":
        selected_indexes = list(range(total_sessions))
    elif history_mode == "selected":
        selected_indexes = list(range(min(8, total_sessions)))
        selected_source_ids = {item["haystack_session_ids"][index] for index in selected_indexes}
        for index, source_id in enumerate(item.get("haystack_session_ids") or []):
            if source_id in answer_ids and source_id not in selected_source_ids:
                selected_indexes.append(index)
                selected_source_ids.add(source_id)
        index = 8
        while len(selected_indexes) < max_sessions and index < total_sessions:
            if index not in selected_indexes:
                selected_indexes.append(index)
            index += 1
        selected_indexes = sorted(selected_indexes)
    else:
        raise ValueError(f"Unsupported MEMORYBENCH_LONGMEMEVAL_HISTORY_MODE: {history_mode}")
    sessions = []
    for source_index in selected_indexes:
        sessions.append(
            {
                "source_session_id": str(item["haystack_session_ids"][source_index]),
                "date": item["haystack_dates"][source_index],
                "turns": item["haystack_sessions"][source_index],
                "is_answer_session": item["haystack_session_ids"][source_index] in answer_ids,
            }
        )
    return {
        "case_id": str(item["question_id"]),
        "question_type": str(item["question_type"]),
        "question": item["question"],
        "question_date": item.get("question_date"),
        "gold_answer": item["answer"],
        "answer_session_ids": list(item.get("answer_session_ids") or []),
        "sessions": sessions,
        "history_mode": history_mode,
        "selected_session_count": len(sessions),
        "total_haystack_sessions": total_sessions,
    }


def session_items(session: dict[str, Any]) -> list[dict[str, str]]:
    items = []
    for idx, turn in enumerate(session["turns"]):
        role = turn.get("role", "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = turn.get("content")
        if isinstance(content, list):
            text = "\n".join(str(part.get("text") or part.get("content") or part) if isinstance(part, dict) else str(part) for part in content)
        elif isinstance(content, dict):
            text = json.dumps(content, ensure_ascii=False)
        else:
            text = "" if content is None else str(content)
        if idx == 0:
            text = f"[Session date: {session['date']}]\n[Source session: {session['source_session_id']}]\n{text}"
        items.append({"type": "message", "role": str(role), "content": text or "[Empty message content.]"})
    return items


def ensure_store(client: AIProjectClient) -> None:
    existing = next((store for store in client.beta.memory_stores.list() if getattr(store, "name", None) == STORE), None)
    if existing:
        return
    client.beta.memory_stores.create(
        name=STORE,
        definition=MemoryStoreDefaultDefinition(
            chat_model=ANSWER_DEPLOYMENT,
            embedding_model=EMBEDDING_DEPLOYMENT,
            options=MemoryStoreDefaultOptions(
                chat_summary_enabled=env_bool("MEMORYBENCH_MEMORY_CHAT_SUMMARY_ENABLED", True),
                user_profile_enabled=env_bool("MEMORYBENCH_MEMORY_USER_PROFILE_ENABLED", False),
                procedural_memory_enabled=env_bool("MEMORYBENCH_MEMORY_PROCEDURAL_ENABLED", False),
                default_ttl_seconds=90 * 24 * 60 * 60,
            ),
        ),
        description="MemoryBench live QA Foundry Memory store",
    )


def wait_for_memory(client: AIProjectClient, scope: str, timeout: int = 240) -> int:
    start = time.time()
    last = -1
    stable = 0
    while time.time() - start < timeout:
        count = len(list(client.beta.memory_stores.list_memories(STORE, scope=scope, limit=500)))
        if count != last:
            log(f"memory scope={scope} count={count}")
        if count > 0 and count == last:
            stable += 1
            if stable >= 2:
                return count
        else:
            stable = 0
            last = count
        time.sleep(5)
    return max(last, 0)


def ingest_case(client: AIProjectClient, case: dict[str, Any], scope: str) -> dict[str, Any]:
    retries = env_int("MEMORYBENCH_MEMORY_INGEST_RETRIES", 3)
    wait_timeout = env_int("MEMORYBENCH_MEMORY_WAIT_TIMEOUT_SECONDS", 240)
    fail_on_partial = env_bool("MEMORYBENCH_FAIL_ON_PARTIAL_MEMORY_INGESTION", False)
    session_delay = env_int("MEMORYBENCH_MEMORY_INGEST_SESSION_DELAY_SECONDS", 0)
    session_results = []
    errors = []
    client.beta.memory_stores.delete_scope(STORE, scope=scope)
    time.sleep(1)
    for session_index, session in enumerate(case["sessions"], start=1):
        log(
            "ingest "
            f"case={case['case_id']} scope={scope} "
            f"session={session_index}/{len(case['sessions'])} source={session['source_session_id']}"
        )
        session_result = {
            "source_session_id": session["source_session_id"],
            "date": session["date"],
            "attempts": 0,
            "succeeded": False,
        }
        last_error = None
        for attempt in range(1, retries + 1):
            session_result["attempts"] = attempt
            try:
                client.beta.memory_stores.begin_update_memories(STORE, scope=scope, items=session_items(session), update_delay=0).result()
                session_result["succeeded"] = True
                break
            except Exception as exc:
                last_error = exc
                details = exception_diagnostics(exc, message_limit=500, body_limit=500)
                if attempt < retries:
                    delay = retry_delay_seconds(exc, attempt)
                    log(
                        "ingest retry "
                        f"case={case['case_id']} scope={scope} "
                        f"session={session_index}/{len(case['sessions'])} attempt={attempt} "
                        f"delay_seconds={delay:.1f} error_details={json.dumps(details, ensure_ascii=False)}"
                    )
                    time.sleep(delay)
        if last_error is not None and not session_result["succeeded"]:
            details = exception_diagnostics(last_error, message_limit=1200, body_limit=1200)
            session_result["error"] = details
            log(
                "ingest failed "
                f"case={case['case_id']} scope={scope} "
                f"session={session_index}/{len(case['sessions'])} attempts={session_result['attempts']} "
                f"error_details={json.dumps(details, ensure_ascii=False)}"
            )
            errors.append(
                {
                    "stage": "memory_ingest",
                    "source_session_id": session["source_session_id"],
                    **details,
                    "content_filter": is_content_filter_error(last_error),
                    "rate_limit": is_rate_limit_error(last_error),
                    "auth_error": is_auth_error(last_error),
                }
            )
        session_results.append(session_result)
        if session_delay > 0 and session_index < len(case["sessions"]):
            time.sleep(session_delay)
    memory_count = wait_for_memory(client, scope, timeout=wait_timeout)
    complete = all(result["succeeded"] for result in session_results) and memory_count > 0
    metadata = {
        "memory_count": memory_count,
        "expected_session_count": len(case["sessions"]),
        "ingested_session_count": sum(1 for result in session_results if result["succeeded"]),
        "complete": complete,
        "session_results": session_results,
        "errors": errors,
    }
    if fail_on_partial and not complete:
        raise RuntimeError(f"Memory ingestion incomplete for {case['case_id']}: {json.dumps(metadata, ensure_ascii=False)[:2000]}")
    return metadata


def retrieval_query(question: str, question_date: str | None) -> str:
    if env_bool("MEMORYBENCH_INCLUDE_QUESTION_DATE_IN_RETRIEVAL_QUERY", False) and question_date:
        return f"[Question date: {question_date}]\n{question}"
    return question


def retrieve_memories(client: AIProjectClient, scope: str, question: str, question_date: str | None = None) -> list[dict[str, Any]]:
    max_memories = env_int("MEMORYBENCH_MEMORY_MAX_MEMORIES", 8)
    response = client.beta.memory_stores.search_memories(
        STORE,
        scope=scope,
        items=[{"type": "message", "role": "user", "content": retrieval_query(question, question_date)}],
        options=MemorySearchOptions(max_memories=max_memories),
    )
    memories = []
    for item in getattr(response, "memories", []):
        if hasattr(item, "as_dict"):
            raw = item.as_dict()
        elif hasattr(item, "model_dump"):
            raw = item.model_dump(mode="json")
        else:
            raw = getattr(item, "__dict__", {"value": str(item)})
        memories.append(raw.get("memory_item", raw))
    return memories


def memory_scope(condition: str, case_id: str) -> str:
    prefix = os.environ.get("MEMORYBENCH_MEMORY_SCOPE_PREFIX")
    max_memories = env_int("MEMORYBENCH_MEMORY_MAX_MEMORIES", 8)
    if not prefix:
        prefix = f"k{max_memories}"
    expected_k = f"k{max_memories}"
    prefix_tokens = prefix.replace("_", "-").split("-")
    if expected_k not in prefix_tokens:
        raise ValueError(
            "MEMORYBENCH_MEMORY_SCOPE_PREFIX must include the configured retrieval budget "
            f"({expected_k}); got {prefix!r}"
        )
    return f"longmemeval-live-{prefix}-{condition}-{case_id}"


def question_block(question: str, question_date: str | None) -> str:
    if question_date:
        return f"Current Date: {question_date}\nQuestion: {question}"
    return f"Question: {question}"


def answer_prompt(question: str, memories: list[dict[str, Any]] | None, question_date: str | None = None) -> list[dict[str, str]]:
    if memories is None:
        context = "No memory context is available for this baseline run."
    elif not memories:
        context = "No relevant memories were retrieved."
    else:
        context = "\n\n".join(f"[Memory {i}] {memory.get('content')}" for i, memory in enumerate(memories, start=1))
    return [
        {
            "role": "system",
            "content": (
                "You are answering a LongMemEval memory question. Use only the provided memory context. "
                "Search the memories carefully before deciding the answer is missing. "
                "For preference questions, infer the user's preference from specific stated likes/dislikes and constraints. "
                "For temporal or knowledge-update questions, prefer newer or more specific facts when evidence conflicts. "
                "If the evidence supports an answer, answer directly and concisely. "
                "Only say you do not know if no provided memory supports the answer."
            ),
        },
        {"role": "user", "content": f"Memory context:\n{context}\n\n{question_block(question, question_date)}"},
    ]


def structured_memory_prompt(question: str, memories: list[dict[str, Any]] | None, question_date: str | None = None) -> list[dict[str, str]]:
    if not memories:
        context = "No relevant memories were retrieved."
    else:
        blocks = []
        for index, memory in enumerate(memories, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Memory {index}",
                        f"kind: {memory.get('kind')}",
                        f"updated_at: {memory.get('updated_at')}",
                        f"scope: {memory.get('scope')}",
                        f"content: {memory.get('content')}",
                    ]
                )
            )
        context = "\n\n---\n\n".join(blocks)
    return [
        {
            "role": "system",
            "content": (
                "You are answering a LongMemEval memory question using retrieved long-term memories. "
                "Inspect every memory for direct or indirect evidence before answering. "
                "For preference questions, answer from the user's stated preferences, constraints, and dislikes. "
                "For temporal reasoning, use dates/timestamps and event order. "
                "For knowledge updates, prefer the latest or most specific memory when facts conflict. "
                "For multi-session questions, combine evidence across memories. "
                "If evidence supports the gold fact in paraphrase form, answer it even when the exact words differ. "
                "Only say you do not know if none of the retrieved memories support the answer. "
                "Return a concise final answer only; do not include analysis."
            ),
        },
        {"role": "user", "content": f"Retrieved memories:\n{context}\n\n{question_block(question, question_date)}"},
    ]


def transcript_context(sessions: list[dict[str, Any]]) -> str:
    chunks = []
    for session in sessions:
        turns = []
        for item in session_items(session):
            turns.append(f"{item['role']}: {item['content']}")
        chunks.append(f"Session date: {session['date']}\nSource session: {session['source_session_id']}\n" + "\n".join(turns))
    return "\n\n---\n\n".join(chunks)


def context_prompt(question: str, context: str, label: str, question_date: str | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are answering a LongMemEval memory question using only the provided {label}. "
                "The answer is often present indirectly in prior user-assistant interaction history. "
                "Search the context carefully. For preference questions, infer the user's preference from stated preferences, constraints, and negative preferences. "
                "For temporal reasoning, use dates and event order. For knowledge updates, prefer the latest or most specific evidence. "
                "For multi-session questions, combine evidence across sessions. "
                "Answer directly and concisely when the context supports an answer. "
                "Only say you do not know if the provided context truly lacks support."
            ),
        },
        {"role": "user", "content": f"{label}:\n{context}\n\n{question_block(question, question_date)}"},
    ]


def answer_sessions(case: dict[str, Any]) -> list[dict[str, Any]]:
    answer_ids = set(case.get("answer_session_ids") or [])
    selected = [session for session in case["sessions"] if session["source_session_id"] in answer_ids or session.get("is_answer_session")]
    return selected or case["sessions"]


def memory_contains_answer(memories: list[dict[str, Any]], answer: str) -> bool:
    answer_text = str(answer or "").strip().lower()
    if not answer_text:
        return False
    return answer_text in json.dumps(memories, ensure_ascii=False).lower()


def judge_prompt(question_type: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if abstention:
        return f"I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    if question_type in {"single-session-user", "single-session-assistant", "multi-session"}:
        return f"I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    if question_type == "temporal-reasoning":
        return f"I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    if question_type == "knowledge-update":
        return f"I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    if question_type == "single-session-preference":
        return f"I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    raise NotImplementedError(f"Unsupported LongMemEval question_type for judging: {question_type}")


def evidence_support_prompt(question: str, answer: str, memories: list[dict[str, Any]]) -> str:
    memory_text = json.dumps(
        [
            {
                "kind": memory.get("kind"),
                "updated_at": memory.get("updated_at"),
                "content": memory.get("content"),
            }
            for memory in memories
        ],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are checking whether retrieved memories contain evidence needed to answer a question.\n"
        "Answer yes if the memories directly or indirectly support the gold answer. "
        "Answer no if the evidence is absent, contradictory, or too vague.\n\n"
        f"Question: {question}\n\nGold answer: {answer}\n\nRetrieved memories:\n{memory_text}\n\n"
        "Do the retrieved memories support the gold answer? Answer yes or no only."
    )


def yes_no(text: str) -> bool:
    return "yes" in (text or "").lower()


def is_content_filter_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "content_filter" in text or "contentfiltered" in text or "content management policy" in text


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "ratelimit" in text or "rate limit" in text or "status_code\":429" in text or "429" in text


def is_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "authentication" in text or "unauthorized" in text or "status_code\":401" in text or " 401" in text


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    base = env_int("MEMORYBENCH_MEMORY_INGEST_RETRY_DELAY_SECONDS", 10)
    rate_limit_base = env_int("MEMORYBENCH_MEMORY_INGEST_RATE_LIMIT_DELAY_SECONDS", 60)
    max_delay = env_int("MEMORYBENCH_MEMORY_INGEST_MAX_RETRY_DELAY_SECONDS", 300)
    selected_base = rate_limit_base if is_rate_limit_error(exc) else base
    jitter = random.uniform(0, min(5, selected_base))
    return min(max_delay, selected_base * (2 ** max(0, attempt - 1)) + jitter)


def complete_chat_with_backoff(client: LLMClient, prompt: Any, *, max_tokens: int, temperature: Any) -> str:
    retries = env_int("MEMORYBENCH_LLM_RETRIES", 4)
    retry_delay = env_float("MEMORYBENCH_LLM_RETRY_DELAY_SECONDS", 20.0)
    rate_limit_delay = env_float("MEMORYBENCH_LLM_RATE_LIMIT_DELAY_SECONDS", 60.0)
    max_delay = env_float("MEMORYBENCH_LLM_MAX_RETRY_DELAY_SECONDS", 240.0)
    call_delay = env_float("MEMORYBENCH_LLM_CALL_DELAY_SECONDS", 2.0)

    for attempt in range(1, retries + 1):
        try:
            response = client.complete_chat(prompt, max_tokens=max_tokens, temperature=temperature)
            if call_delay > 0:
                time.sleep(call_delay)
            return response
        except Exception as exc:
            if attempt >= retries:
                raise
            selected_delay = rate_limit_delay if is_rate_limit_error(exc) else retry_delay
            delay = min(max_delay, selected_delay * (2 ** max(0, attempt - 1)) + random.uniform(0, min(5, selected_delay)))
            log(
                "llm retry "
                f"attempt={attempt}/{retries} delay_seconds={delay:.1f} "
                f"error_details={json.dumps(exception_diagnostics(exc), ensure_ascii=False)}"
            )
            time.sleep(delay)

    raise RuntimeError("unreachable LLM retry state")


def run_condition(
    cases: list[dict[str, Any]],
    condition: str,
    client: AIProjectClient,
    answer_client: LLMClient,
    judge_client: LLMClient,
    out_dir: Path,
    memory_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results = []
    log(f"condition={condition} start case_count={len(cases)}")
    for case_index, case in enumerate(cases, start=1):
        log(f"condition={condition} case={case_index}/{len(cases)} id={case['case_id']} start")
        scope = memory_scope(condition, case["case_id"])
        memories = None
        memory_count = 0
        memory_ingestion = None
        memory_reused = False
        full_transcript_context = None
        full_transcript_sessions = None
        errors = []
        if condition in {"foundry-memory", "foundry-memory-structured"}:
            try:
                cache_key = case["case_id"]
                if memory_cache is not None and cache_key in memory_cache:
                    cached = memory_cache[cache_key]
                    memory_ingestion = cached["memory_ingestion"]
                    memory_count = cached["memory_count"]
                    memories = cached["memories"]
                    scope = cached["scope"]
                    memory_reused = True
                    log(
                        "memory reuse "
                        f"condition={condition} case={case['case_id']} scope={scope} "
                        f"retrieved={len(memories)}"
                    )
                else:
                    if memory_cache is not None:
                        scope = memory_scope("foundry-memory", case["case_id"])
                    memory_ingestion = ingest_case(client, case, scope)
                    memory_count = memory_ingestion["memory_count"]
                    errors.extend(memory_ingestion["errors"])
                    memories = retrieve_memories(client, scope, case["question"], case.get("question_date"))
                    if memory_cache is not None:
                        memory_cache[cache_key] = {
                            "memory_ingestion": memory_ingestion,
                            "memory_count": memory_count,
                            "memories": memories,
                            "scope": scope,
                        }
            except Exception as exc:
                memories = []
                errors.append({"stage": "memory", "type": type(exc).__name__, "message": str(exc)[:1200]})
            prompt = structured_memory_prompt(case["question"], memories, case.get("question_date")) if condition == "foundry-memory-structured" else answer_prompt(case["question"], memories, case.get("question_date"))
        elif condition in {"full-transcript", "selected-transcript"}:
            full_transcript_context = transcript_context(case["sessions"])
            full_transcript_sessions = [
                {
                    "source_session_id": session["source_session_id"],
                    "date": session["date"],
                    "is_answer_session": session.get("is_answer_session", False),
                    "turn_count": len(session.get("turns") or []),
                }
                for session in case["sessions"]
            ]
            prompt = context_prompt(case["question"], full_transcript_context, "full transcript context", case.get("question_date"))
        elif condition == "oracle-context":
            prompt = context_prompt(case["question"], transcript_context(answer_sessions(case)), "oracle answer-session context", case.get("question_date"))
        elif condition == "question-only":
            prompt = answer_prompt(case["question"], None, case.get("question_date"))
        else:
            raise ValueError(f"Unsupported LongMemEval condition: {condition}")
        try:
            response = complete_chat_with_backoff(answer_client, prompt, max_tokens=512, temperature=None)
        except Exception as exc:
            response = ""
            errors.append({"stage": "answer", "type": type(exc).__name__, "content_filter": is_content_filter_error(exc), "message": str(exc)[:1000]})
        try:
            judge = complete_chat_with_backoff(
                judge_client,
                [{"role": "user", "content": judge_prompt(case["question_type"], case["question"], case["gold_answer"], response, "_abs" in case["case_id"])}],
                max_tokens=32,
                temperature=None,
            )
            correct = yes_no(judge)
        except Exception as exc:
            judge = ""
            correct = False
            errors.append({"stage": "judge", "type": type(exc).__name__, "content_filter": is_content_filter_error(exc), "message": str(exc)[:1000]})
        evidence_judge = ""
        evidence_supported = False
        if memories is not None:
            try:
                evidence_judge = complete_chat_with_backoff(
                    judge_client,
                    [{"role": "user", "content": evidence_support_prompt(case["question"], case["gold_answer"], memories)}],
                    max_tokens=32,
                    temperature=None,
                )
                evidence_supported = yes_no(evidence_judge)
            except Exception as exc:
                errors.append({"stage": "evidence_judge", "type": type(exc).__name__, "content_filter": is_content_filter_error(exc), "message": str(exc)[:1000]})
        result = {
            "condition": condition,
            "case_id": case["case_id"],
            "question_type": case["question_type"],
            "question": case["question"],
            "question_date": case.get("question_date"),
            "gold_answer": case["gold_answer"],
            "response": response,
            "judge": judge,
            "answer_correct": correct,
            "retrieved_memory_count": 0 if memories is None else len(memories),
            "retrieved_memories": memories,
            "retrieved_memory_contains_gold_answer": False if memories is None else memory_contains_answer(memories, case["gold_answer"]),
            "retrieved_memory_evidence_judge": evidence_judge,
            "retrieved_memory_evidence_supported": evidence_supported,
            "memory_count": memory_count,
            "memory_ingestion": memory_ingestion,
            "memory_ingestion_complete": True if memory_ingestion is None else memory_ingestion["complete"],
            "memory_max_memories": env_int("MEMORYBENCH_MEMORY_MAX_MEMORIES", 8) if memories is not None else None,
            "memory_scope": scope if memories is not None else None,
            "memory_reused": memory_reused,
            "full_transcript_session_count": None if full_transcript_sessions is None else len(full_transcript_sessions),
            "full_transcript_sessions": full_transcript_sessions,
            "full_transcript_context": full_transcript_context,
            "full_transcript_context_char_count": None if full_transcript_context is None else len(full_transcript_context),
            "full_transcript_context_rough_token_count": None if full_transcript_context is None else len(full_transcript_context) // 4,
            "selected_transcript_session_count": None if full_transcript_sessions is None else len(full_transcript_sessions),
            "selected_transcript_sessions": full_transcript_sessions,
            "selected_transcript_context": full_transcript_context,
            "errors": errors,
        }
        results.append(result)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{condition}-{case['case_id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(
            f"condition={condition} case={case_index}/{len(cases)} id={case['case_id']} "
            f"correct={correct} errors={len(errors)} retrieved={result['retrieved_memory_count']}"
        )
    log(f"condition={condition} done")
    return results


def log_metrics(metrics: dict[str, float], artifacts: list[Path]) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping AML logging.")
        return
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        mlflow.set_tag("memorybench.phase", "phase1")
        mlflow.set_tag("memorybench.benchmark_id", "longmemeval-s")
        mlflow.set_tag("memorybench.metric_type", "live_golden_answer_correctness")
        logged_metrics = (
            metrics
            if env_bool("MEMORYBENCH_LOG_ALL_MLFLOW_METRICS", False)
            else {key: value for key, value in metrics.items() if key in PRIMARY_MLFLOW_METRICS}
        )
        for key, value in logged_metrics.items():
            mlflow.log_metric(key, value)
            mlflow.log_metric(f"{key}_series", value, step=0)
            mlflow.log_metric(f"{key}_series", value, step=1)
        for path in artifacts:
            try:
                mlflow.log_artifact(str(path), artifact_path="memorybench")
            except Exception as exc:
                print(f"Failed to log MLflow artifact {path}: {type(exc).__name__}: {str(exc)[:500]}")


def condition_accuracy(results: list[dict[str, Any]]) -> float:
    return sum(1 for result in results if result["answer_correct"]) / len(results) if results else 0.0


def selected_conditions() -> set[str]:
    configured = os.environ.get("MEMORYBENCH_CONDITIONS", "all").strip().lower()
    all_conditions = {
        "question-only",
        "full-transcript",
        "oracle-context",
        "foundry-memory",
        "foundry-memory-structured",
    }
    if not configured or configured == "all":
        return all_conditions
    aliases = {
        "baseline": "question-only",
        "selected-transcript": "full-transcript",
        "memory": "foundry-memory",
        "memory-structured": "foundry-memory-structured",
    }
    requested = {aliases.get(item.strip(), item.strip()) for item in configured.split(",") if item.strip()}
    unknown = requested - all_conditions
    if unknown:
        raise ValueError(f"Unsupported MEMORYBENCH_CONDITIONS value(s): {', '.join(sorted(unknown))}")
    return requested


def clean_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if not result.get("errors") and result.get("memory_ingestion_complete", True)]


def safe_accuracy(results: list[dict[str, Any]]) -> float:
    return sum(1 for result in results if result["answer_correct"]) / len(results) if results else 0.0


def matched_clean_memory_cases(memory: list[dict[str, Any]], memory_structured: list[dict[str, Any]]) -> set[str]:
    clean_memory = {result["case_id"] for result in clean_results(memory)}
    clean_structured = {result["case_id"] for result in clean_results(memory_structured)}
    return clean_memory & clean_structured


def case_accuracy_on_ids(results: list[dict[str, Any]], case_ids: set[str]) -> float:
    subset = [result for result in results if result["case_id"] in case_ids]
    return safe_accuracy(subset)


def shared_clean_case_ids(*conditions: list[dict[str, Any]]) -> set[str]:
    clean_id_sets = [{result["case_id"] for result in clean_results(condition)} for condition in conditions]
    if not clean_id_sets:
        return set()
    return set.intersection(*clean_id_sets)


def memory_diagnostics(memory: list[dict[str, Any]], memory_structured: list[dict[str, Any]]) -> dict[str, float]:
    clean_memory = clean_results(memory)
    matched_clean = matched_clean_memory_cases(memory, memory_structured)
    retrieved_answer = [result for result in memory if result["retrieved_memory_contains_gold_answer"]]
    evidence_supported = [result for result in memory if result["retrieved_memory_evidence_supported"]]
    correct_and_retrieved = [
        result for result in memory
        if result["answer_correct"] and result["retrieved_memory_contains_gold_answer"]
    ]
    correct_and_supported = [
        result for result in memory
        if result["answer_correct"] and result["retrieved_memory_evidence_supported"]
    ]
    return {
        "clean_case_count": float(len(clean_memory)),
        "clean_foundry_memory_answer_accuracy": safe_accuracy(clean_memory),
        "clean_foundry_memory_percent_correct": safe_accuracy(clean_memory) * 100,
        "matched_clean_memory_case_count": float(len(matched_clean)),
        "memory_ingestion_incomplete_count": float(sum(1 for result in memory if not result.get("memory_ingestion_complete", True))),
        "memory_cases_with_errors_count": float(sum(1 for result in memory if result.get("errors"))),
        "memory_retrieval_recall_substring": len(retrieved_answer) / len(memory) if memory else 0.0,
        "memory_retrieval_recall_substring_percent": (len(retrieved_answer) / len(memory) * 100) if memory else 0.0,
        "memory_retrieval_recall_evidence_judge": len(evidence_supported) / len(memory) if memory else 0.0,
        "memory_retrieval_recall_evidence_judge_percent": (len(evidence_supported) / len(memory) * 100) if memory else 0.0,
        "memory_answer_correct_given_substring_retrieved": safe_accuracy(retrieved_answer),
        "memory_answer_correct_given_evidence_supported": safe_accuracy(evidence_supported),
        "memory_correct_and_substring_retrieved_count": float(len(correct_and_retrieved)),
        "memory_correct_and_evidence_supported_count": float(len(correct_and_supported)),
    }


def clean_comparison_diagnostics(
    baseline: list[dict[str, Any]],
    selected_context: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    memory: list[dict[str, Any]],
    memory_structured: list[dict[str, Any]],
) -> dict[str, float]:
    case_ids = shared_clean_case_ids(baseline, selected_context, oracle, memory, memory_structured)
    baseline_acc = case_accuracy_on_ids(baseline, case_ids)
    selected_acc = case_accuracy_on_ids(selected_context, case_ids)
    oracle_acc = case_accuracy_on_ids(oracle, case_ids)
    memory_acc = case_accuracy_on_ids(memory, case_ids)
    structured_acc = case_accuracy_on_ids(memory_structured, case_ids)
    return {
        "shared_clean_case_count": float(len(case_ids)),
        "shared_clean_question_only_percent_correct": baseline_acc * 100,
        "shared_clean_selected_transcript_percent_correct": selected_acc * 100,
        "shared_clean_oracle_context_percent_correct": oracle_acc * 100,
        "shared_clean_foundry_memory_percent_correct": memory_acc * 100,
        "shared_clean_foundry_memory_structured_percent_correct": structured_acc * 100,
        "shared_clean_memory_lift_pp": (memory_acc - baseline_acc) * 100,
        "shared_clean_memory_vs_selected_transcript_delta_pp": (memory_acc - selected_acc) * 100,
        "shared_clean_memory_vs_oracle_gap_pp": (memory_acc - oracle_acc) * 100,
    }


def timestamp_fidelity_diagnostics(cases: list[dict[str, Any]]) -> dict[str, float]:
    cases_with_question_date = sum(1 for case in cases if case.get("question_date"))
    sessions = [session for case in cases for session in case["sessions"]]
    sessions_with_dates = sum(1 for session in sessions if session.get("date"))
    return {
        "question_date_present_count": float(cases_with_question_date),
        "question_date_present_percent": (cases_with_question_date / len(cases) * 100) if cases else 0.0,
        "session_date_present_count": float(sessions_with_dates),
        "session_date_present_percent": (sessions_with_dates / len(sessions) * 100) if sessions else 0.0,
    }


def validity_report(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    metrics: dict[str, float],
    history_mode: str,
) -> dict[str, Any]:
    checks = [
        {
            "checkId": "benchmark_set_hash",
            "label": "Benchmark set hash",
            "status": "passed" if manifest.get("datasetHash") else "failed",
            "message": f"datasetHash={manifest.get('datasetHash')}" if manifest.get("datasetHash") else "Missing datasetHash.",
        },
        {
            "checkId": "paper_faithful_history_mode",
            "label": f"{benchmark_label(manifest)} full-history protocol",
            "status": "passed" if history_mode == "full" else "failed",
            "message": f"historyMode={history_mode}; official runs must use full haystack history.",
        },
        {
            "checkId": "timestamp_fidelity",
            "label": "Question/session timestamp fidelity",
            "status": "passed"
            if metrics.get("question_date_present_percent") == 100.0 and metrics.get("session_date_present_percent") == 100.0
            else "failed",
            "message": (
                f"question_date_present_percent={metrics.get('question_date_present_percent')}; "
                f"session_date_present_percent={metrics.get('session_date_present_percent')}"
            ),
        },
        {
            "checkId": "bare_question_retrieval",
            "label": "Bare-question retrieval protocol",
            "status": "passed" if not env_bool("MEMORYBENCH_INCLUDE_QUESTION_DATE_IN_RETRIEVAL_QUERY", False) else "warning",
            "message": "Foundry Memory retrieval uses the bare question; Current Date is supplied only to answer prompts.",
        },
        {
            "checkId": "judge_prompt_versioning",
            "label": "Judge and prompt versioning",
            "status": "passed",
            "message": (
                f"answer={ANSWER_PROMPT_VERSION}; judge={JUDGE_PROMPT_VERSION}; "
                f"evidenceJudge={EVIDENCE_JUDGE_PROMPT_VERSION}"
            ),
        },
        {
            "checkId": "memory_ingestion_complete",
            "label": "Memory ingestion completeness",
            "status": "passed" if metrics.get("memory_ingestion_incomplete_count", 0.0) == 0.0 else "warning",
            "message": f"incomplete_memory_cases={metrics.get('memory_ingestion_incomplete_count', 0.0)}",
        },
        {
            "checkId": "memory_snapshot",
            "label": "Memory snapshot/hash",
            "status": "warning",
            "message": "Foundry Memory snapshot/hash capture is not available yet; scope/config/provenance are recorded instead.",
        },
    ]
    return {
        "schemaVersion": "0.1.0",
        "benchmark": benchmark_label(manifest),
        "benchmarkSet": manifest.get("benchmarkSet"),
        "caseCount": len(cases),
        "validityStatus": "invalid"
        if any(check["status"] == "failed" for check in checks)
        else "warning"
        if any(check["status"] == "warning" for check in checks)
        else "valid",
        "checks": checks,
        "generatedAt": now_iso(),
    }


def reproducibility_report(
    manifest: dict[str, Any],
    history_mode: str,
    shard_index: str | None,
    shard_count: str | None,
) -> dict[str, Any]:
    return {
        "schemaVersion": "0.1.0",
        "benchmark": benchmark_label(manifest),
        "benchmarkSet": manifest.get("benchmarkSet"),
        "datasetHash": manifest.get("datasetHash"),
        "sourceSha256": manifest.get("sourceSha256"),
        "historyMode": history_mode,
        "shardIndex": None if shard_index is None else int(shard_index),
        "shardCount": None if shard_count is None else int(shard_count),
        "memoryStore": STORE,
        "memoryScopePrefix": os.environ.get("MEMORYBENCH_MEMORY_SCOPE_PREFIX") or f"k{env_int('MEMORYBENCH_MEMORY_MAX_MEMORIES', 8)}",
        "memoryMaxMemories": env_int("MEMORYBENCH_MEMORY_MAX_MEMORIES", 8),
        "reuseMemory": env_bool("MEMORYBENCH_REUSE_MEMORY", True),
        "includeQuestionDateInRetrievalQuery": env_bool("MEMORYBENCH_INCLUDE_QUESTION_DATE_IN_RETRIEVAL_QUERY", False),
        "answerPromptVersion": ANSWER_PROMPT_VERSION,
        "judgePromptVersion": JUDGE_PROMPT_VERSION,
        "evidenceJudgePromptVersion": EVIDENCE_JUDGE_PROMPT_VERSION,
        "answerDeployment": ANSWER_DEPLOYMENT,
        "judgeDeployment": JUDGE_DEPLOYMENT,
        "embeddingDeployment": EMBEDDING_DEPLOYMENT,
        "generatedAt": now_iso(),
    }


def validate_timestamp_fidelity(cases: list[dict[str, Any]]) -> None:
    missing_question_dates = [case["case_id"] for case in cases if not case.get("question_date")]
    missing_session_dates = [
        f"{case['case_id']}:{session['source_session_id']}"
        for case in cases
        for session in case["sessions"]
        if not session.get("date")
    ]
    if missing_question_dates or missing_session_dates:
        raise ValueError(
            "LongMemEval timestamp fidelity check failed: "
            f"missing_question_dates={missing_question_dates[:10]}, "
            f"missing_session_dates={missing_session_dates[:10]}"
        )


def main() -> None:
    root = Path(os.environ.get("MEMORYBENCH_BENCHMARK_SET_DIR", "benchmark_sets/longmemeval-s-20-raw-v1"))
    out_dir = Path(os.environ.get("MEMORYBENCH_OUTPUT_DIR", "outputs"))
    manifest, raw_cases = load_cases(root)
    case_limit = int(os.environ.get("MEMORYBENCH_LONGMEMEVAL_CASE_LIMIT", "20"))
    history_mode = os.environ.get("MEMORYBENCH_LONGMEMEVAL_HISTORY_MODE", "full").strip().lower()
    shard_index = os.environ.get("MEMORYBENCH_SHARD_INDEX")
    shard_count = os.environ.get("MEMORYBENCH_SHARD_COUNT")
    selected_raw_cases = raw_cases[:case_limit]
    if shard_index is not None or shard_count is not None:
        if shard_index is None or shard_count is None:
            raise ValueError("MEMORYBENCH_SHARD_INDEX and MEMORYBENCH_SHARD_COUNT must be provided together")
        shard_index_int = int(shard_index)
        shard_count_int = int(shard_count)
        if shard_count_int < 1 or shard_index_int < 0 or shard_index_int >= shard_count_int:
            raise ValueError("Invalid shard index/count")
        selected_raw_cases = [case for index, case in enumerate(selected_raw_cases) if index % shard_count_int == shard_index_int]
    cases = [normalize_case(item, history_mode=history_mode) for item in selected_raw_cases]
    validate_timestamp_fidelity(cases)
    reuse_memory = env_bool("MEMORYBENCH_REUSE_MEMORY", True)
    log(
        "LongMemEval live run start "
        f"history_mode={history_mode} shard={shard_index}/{shard_count} "
        f"cases={len(cases)} max_memories={env_int('MEMORYBENCH_MEMORY_MAX_MEMORIES', 8)} "
        f"reuse_memory={reuse_memory}"
    )
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
    ensure_store(project_client)
    answer_client = LLMClient(endpoint=OPENAI_ENDPOINT, deployment=ANSWER_DEPLOYMENT)
    judge_client = LLMClient(endpoint=OPENAI_ENDPOINT, deployment=JUDGE_DEPLOYMENT)

    memory_cache: dict[str, dict[str, Any]] | None = {} if reuse_memory else None
    conditions = selected_conditions()
    log(f"conditions={','.join(sorted(conditions))}")
    baseline = (
        run_condition(cases, "question-only", project_client, answer_client, judge_client, out_dir / "cases")
        if "question-only" in conditions
        else []
    )
    selected_context = (
        run_condition(cases, "full-transcript", project_client, answer_client, judge_client, out_dir / "cases")
        if "full-transcript" in conditions
        else []
    )
    oracle = (
        run_condition(cases, "oracle-context", project_client, answer_client, judge_client, out_dir / "cases")
        if "oracle-context" in conditions
        else []
    )
    memory = (
        run_condition(cases, "foundry-memory", project_client, answer_client, judge_client, out_dir / "cases", memory_cache=memory_cache)
        if "foundry-memory" in conditions
        else []
    )
    memory_structured = (
        run_condition(
            cases,
            "foundry-memory-structured",
            project_client,
            answer_client,
            judge_client,
            out_dir / "cases",
            memory_cache=memory_cache,
        )
        if "foundry-memory-structured" in conditions
        else []
    )
    baseline_acc = condition_accuracy(baseline)
    selected_context_acc = condition_accuracy(selected_context)
    oracle_acc = condition_accuracy(oracle)
    memory_acc = condition_accuracy(memory)
    memory_structured_acc = condition_accuracy(memory_structured)
    metrics = {
        "baseline_answer_accuracy": baseline_acc,
        "baseline_percent_correct": baseline_acc * 100,
        "question_only_answer_accuracy": baseline_acc,
        "question_only_percent_correct": baseline_acc * 100,
        "selected_transcript_answer_accuracy": selected_context_acc,
        "selected_transcript_percent_correct": selected_context_acc * 100,
        "full_transcript_answer_accuracy": selected_context_acc,
        "full_transcript_percent_correct": selected_context_acc * 100,
        "oracle_context_answer_accuracy": oracle_acc,
        "oracle_context_percent_correct": oracle_acc * 100,
        "foundry_memory_answer_accuracy": memory_acc,
        "foundry_memory_percent_correct": memory_acc * 100,
        "foundry_memory_structured_answer_accuracy": memory_structured_acc,
        "foundry_memory_structured_percent_correct": memory_structured_acc * 100,
        "answer_accuracy_lift_pp": (memory_acc - baseline_acc) * 100,
        "structured_answer_accuracy_lift_pp": (memory_structured_acc - baseline_acc) * 100,
        "structured_vs_unstructured_delta_pp": (memory_structured_acc - memory_acc) * 100,
        "memory_vs_selected_transcript_delta_pp": (memory_acc - selected_context_acc) * 100,
        "memory_vs_oracle_gap_pp": (memory_acc - oracle_acc) * 100,
        "structured_memory_vs_selected_transcript_delta_pp": (memory_structured_acc - selected_context_acc) * 100,
        "structured_memory_vs_oracle_gap_pp": (memory_structured_acc - oracle_acc) * 100,
        "case_count": float(len(cases)),
        "memory_retrieval_count": float(sum(result["retrieved_memory_count"] for result in memory)),
        "memory_retrieval_contains_answer_count": float(sum(1 for result in memory if result["retrieved_memory_contains_gold_answer"])),
        "memory_retrieval_evidence_supported_count": float(sum(1 for result in memory if result["retrieved_memory_evidence_supported"])),
        "structured_retrieval_evidence_supported_count": float(sum(1 for result in memory_structured if result["retrieved_memory_evidence_supported"])),
        "error_count": float(sum(len(result.get("errors", [])) for result in [*baseline, *selected_context, *oracle, *memory, *memory_structured])),
        "content_filter_error_count": float(sum(1 for result in [*baseline, *selected_context, *oracle, *memory, *memory_structured] for error in result.get("errors", []) if error.get("content_filter"))),
        "memory_max_memories": float(env_int("MEMORYBENCH_MEMORY_MAX_MEMORIES", 8)),
        "memory_reuse_enabled": float(reuse_memory),
        **memory_diagnostics(memory, memory_structured),
        **clean_comparison_diagnostics(baseline, selected_context, oracle, memory, memory_structured),
        **timestamp_fidelity_diagnostics(cases),
    }
    summary = {
        "schemaVersion": "0.1.0",
        "benchmark": benchmark_label(manifest),
        "benchmarkSet": manifest.get("benchmarkSet"),
        "datasetHash": manifest.get("datasetHash"),
        "historyMode": history_mode,
        "conditions": sorted(conditions),
        "reuseMemory": reuse_memory,
        "shardIndex": None if shard_index is None else int(shard_index),
        "shardCount": None if shard_count is None else int(shard_count),
        "metric": "answer_correctness",
        "promptVersions": {
            "answer": ANSWER_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
            "evidenceJudge": EVIDENCE_JUDGE_PROMPT_VERSION,
        },
        "modelDeployments": {
            "answer": ANSWER_DEPLOYMENT,
            "judge": JUDGE_DEPLOYMENT,
            "embedding": EMBEDDING_DEPLOYMENT,
        },
        "metrics": metrics,
        "baseline": baseline,
        "selectedTranscript": selected_context,
        "fullTranscript": selected_context,
        "oracleContext": oracle,
        "foundryMemory": memory,
        "foundryMemoryStructured": memory_structured,
        "generatedAt": now_iso(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = artifact_slug(benchmark_label(manifest))
    summary_path = out_dir / f"{slug}-live-summary.json"
    metrics_path = out_dir / "mlflow-metrics.json"
    html_path = out_dir / f"{slug}-live-summary.html"
    validity_path = out_dir / "validity.json"
    reproducibility_path = out_dir / "reproducibility.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    validity_path.write_text(json.dumps(validity_report(manifest, cases, metrics, history_mode), indent=2, ensure_ascii=False), encoding="utf-8")
    reproducibility_path.write_text(
        json.dumps(reproducibility_report(manifest, history_mode, shard_index, shard_count), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path.write_text(
        f"<!doctype html><html><body><h1>{benchmark_label(manifest)} live sample</h1><p>Baseline: {baseline_acc*100:.1f}%</p><p>Foundry Memory: {memory_acc*100:.1f}%</p><p>Lift: {(memory_acc-baseline_acc)*100:+.1f} pp</p></body></html>",
        encoding="utf-8",
    )
    log_metrics(metrics, [summary_path, metrics_path, html_path, validity_path, reproducibility_path])
    print(json.dumps({"metrics": metrics, "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
