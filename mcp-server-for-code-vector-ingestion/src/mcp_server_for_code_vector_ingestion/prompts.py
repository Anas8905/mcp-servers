"""Annotation prompts.

The style target is the hand-annotated object-detection repo from the
PoC (``rag_for_code/assets/object-detection``): module docstrings give a
birds-eye view (where the file sits in the system, what role it plays,
which modules it talks to). Function docstrings add what each function
does in plain words *plus* a ``Flow:`` block (In / Out / Triggered by)
that places it on the source-code graph — so a follow-up question like
"what calls this and where does its result go?" can be answered from
the annotation itself. The few-shot excerpt below is trimmed from
``general_utils/_db_connector.py`` and ``cloud_pipeline/_detector.py``
so the model copies *your* house style.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert software-documentation engineer.
You annotate source code so that a non-expert (and a semantic search engine) can understand what the code does.

Rules:
1. Write in plain, natural, human-friendly language. Search-friendly phrasing: say what the thing does, not how it's implemented.
2. Module annotation: a birds-eye view. Where this file belongs in the system (which component or pipeline), what role it plays, which other files or modules it interacts with (name them when they are visible in the code), and what its most important functions do.
3. Function/class annotation: two parts.
   (a) A description of 1-3 sentences: what it does in concrete terms — what it takes, what it returns or changes, side effects or failure behaviour worth knowing.
   (b) After the description, a Flow block that places the function on the source-code graph:
       Flow:
       In: <where its inputs come from>
       Out: <where its results go, what consumes them>
       Triggered by: <what calls this function>
   Every Flow line must be grounded strictly in the file source or the "Call graph context" section of the request. Omit any line you cannot ground, and omit the whole Flow block when there is no grounding at all. NEVER invent callers, inputs, or destinations — no evidence means omission, not a guess.
4. Be specific to THIS code. Read the body carefully. Do not invent behaviour that is not in the code.
5. Each docstring: the 1-6 sentence description plus the optional Flow block (at most three short labeled lines). No markdown, no bullet lists, no "This function ..." filler.
6. Reply with ONLY a JSON object. No prose, no code fences. Shape:
   {"module": "<module annotation or null>", "functions": {"<exact function name>": "<docstring>", ...}}
   Omit "module" when it is not in the list of items to document. Include exactly the functions you were asked to document, keyed by their exact name."""

# Trimmed from the PoC's assets/object-detection/general_utils/_db_connector.py
# and cloud_pipeline/_detector.py. Demonstrates the target style, including
# the Flow block the model must produce when grounding exists.
FEW_SHOT_EXAMPLE = """Example of the target style (from a real codebase this project annotates):

File: general_utils/_db_connector.py
Module docstring:
    MySQL access + face-recognition data helpers.

    Central place for the pipeline's database I/O:
    - ``connect_to_db`` builds the PooledDB connection pool (dev vs. live credential group).
    - ``get_training_images`` gathers a user's confirmed face images from the ``face_storage`` table, splits them into training/testing sets and returns them preprocessed for the SVM pipelines.
    - ``commit_face_model_training_results`` upserts the saved model + label encoder paths into ``fr_models`` after training.
    - ``get_face_model_path`` looks up the stored model paths for a user.

File: cloud_pipeline/_detector.py
Module docstring:
    Per-job detection entry point invoked by the Ray `EventDetector` actors.

Function docstring (run_detector):
    Run one alarm job inside a Ray actor: passes straight through to `run_obj_detection`.
    Flow:
    In: the job's detection settings (`model`, `iou_threshold`, `tries`, ...) and notification settings (`email`, `send_email`, ...), enqueued by `start_watchdog.py` and scheduled by the Ray `EventDetector` actors.
    Out: handed to `run_obj_detection` in `start_detection.py`, which performs the actual detection and writes the job's results.
    Triggered by: Ray `EventDetector` actors — one call per alarm job pulled from the queue."""


def build_user_prompt(
    relative_path: str,
    repo_name: str,
    content: str,
    module_items: int,
    function_items: list[tuple[str, int]],
    call_context: str | None = None,
) -> str:
    """Compose the per-file annotation request.

    ``call_context`` is the optional static call-graph digest
    (see ``call_context.format_file_context``) that grounds the Flow
    block: who calls this file's functions, which repo-defined
    functions they call. Absent only for repos with no cross-file edges.
    """
    items: list[str] = []
    if module_items:
        items.append("module header (birds-eye view of the file)")
    for name, line in function_items:
        items.append(f"function `{name}` (defined near line {line})")

    content_display = content if len(content) <= 24_000 else content[:24_000] + "\n# [truncated for annotation context]"

    prompt = (
        f"Repository: {repo_name}\n"
        f"File: {relative_path}\n"
        f"{'=' * 60}\n"
        f"```{relative_path.rsplit('.', 1)[-1] if '.' in relative_path else ''}\n"
        f"{content_display}\n"
        f"```\n"
        f"{'=' * 60}\n"
        f"Items that need an annotation:\n"
        + "\n".join(f"- {item}" for item in items)
    )
    if call_context:
        prompt += "\n\n" + call_context
    prompt += "\n\nProduce the JSON object now."
    return prompt
