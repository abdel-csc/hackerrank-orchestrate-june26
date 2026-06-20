"""
Multi-Modal Evidence Review — main pipeline entry point.

Usage:
    python code/main.py                          # uses dataset/claims.csv → output.csv
    python code/main.py --claims PATH --out PATH
"""

import argparse
import base64
import csv
import datetime
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQ_CSV = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV = REPO_ROOT / "output.csv"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ---------------------------------------------------------------------------
# Allowed value sets (kept here so prompt and validation share one source)
# ---------------------------------------------------------------------------

ISSUE_TYPES = (
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
)

OBJECT_PARTS = {
    "car":     ("front_bumper", "rear_bumper", "door", "hood", "windshield",
                "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
                "body", "unknown"),
    "laptop":  ("screen", "keyboard", "trackpad", "hinge", "lid", "corner",
                "port", "base", "body", "unknown"),
    "package": ("box", "package_corner", "package_side", "seal", "label",
                "contents", "item", "unknown"),
}

RISK_FLAGS = (
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
)

SEVERITIES = ("none", "low", "medium", "high", "unknown")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_dotenv() -> None:
    """Read .env from repo root into os.environ without overwriting existing vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def get_vlm_client(strategy: str = "B"):
    """
    Return (openai.OpenAI client, model_id).

    strategy "A" → Groq (meta-llama/llama-4-scout-17b-16e-instruct)
    strategy "B" → OpenRouter (google/gemma-4-31b-it:free)
    """
    from openai import OpenAI

    strategy = strategy.upper()

    if strategy == "A":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Strategy A requires GROQ_API_KEY in .env")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        model = os.getenv("VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        return client, model

    if strategy == "B":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Strategy B requires OPENROUTER_API_KEY in .env")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        model = os.getenv("VLM_MODEL", "google/gemma-4-31b-it:free")
        return client, model

    raise RuntimeError(f"Unknown strategy '{strategy}'. Use A (Groq) or B (OpenRouter).")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_claims(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["user_id"]: row for row in csv.DictReader(f)}


def load_evidence_requirements(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resolve_images(image_paths_str: str) -> list[Path]:
    """
    Split the semicolon-delimited image_paths field and resolve each entry
    to an absolute path under dataset/.

    CSV values look like: images/test/case_001/img_1.jpg
    Actual location:       dataset/images/test/case_001/img_1.jpg
    """
    resolved = []
    for raw in image_paths_str.split(";"):
        raw = raw.strip()
        if raw:
            resolved.append(DATASET_DIR / raw)
    return resolved


def image_id(path: Path) -> str:
    return path.stem


MAX_IMAGE_PX = 2000  # longest side cap before sending to Groq


def preprocess_image(path: Path) -> tuple[str, str] | None:
    """
    Load, resize (cap longest side at MAX_IMAGE_PX), re-encode as JPEG.
    Returns (base64_str, mime_type) or None if the image cannot be processed.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_IMAGE_PX:
                scale = MAX_IMAGE_PX / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return b64, "image/jpeg"
    except Exception as exc:
        print(f"\n  [WARN] Cannot preprocess {path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Evidence requirement lookup
# ---------------------------------------------------------------------------

def applicable_requirements(requirements: list[dict], claim_object: str) -> list[dict]:
    return [r for r in requirements if r["claim_object"] in (claim_object, "all")]


# ---------------------------------------------------------------------------
# VLM prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an insurance damage claim evidence reviewer. Your job is to verify whether \
submitted images actually support what the user claims — not to take the user's word for it.

Given one or more images of a damaged object, a support conversation, user history,
and evidence requirements, produce a structured JSON verdict.

Core rules:
- Images are the primary source of truth. Analyze them carefully before deciding.
- The conversation defines what part and issue to look for.
- User history adds risk context but cannot override clear visual evidence on its own.
- Reference image IDs (e.g. img_1, img_2) in justifications when helpful.
- Return ONLY a valid JSON object — no markdown fences, no text outside the JSON.

--- CLAIM STATUS — apply strictly in this order ---

1. BEFORE defaulting to "supported", actively look for evidence that CONTRADICTS
   the claim:
   - Is the wrong object or wrong part visible (e.g. different car, wrong panel)?
   - Is the claimed damage type absent where the user says it should be?
   - Does the visible damage clearly not match the description (e.g. user says dent,
     image shows a clean surface; user says scratch, image shows intact paint)?
   - Is the damage on a completely different part than claimed?
   If any of these are true, set claim_status to "contradicted".

2. "supported" requires the image to ACTIVELY CONFIRM the claim — the exact damage
   type must be visible on the exact part described. Absence of contradiction is NOT
   sufficient for "supported".

3. Use "not_enough_information" when the image is present and plausible but the
   relevant part or damage cannot be clearly assessed (e.g. wrong angle, too blurry,
   part not in frame).

--- SEVERITY ---

"none" means confirmed no damage is visible on the relevant part. "unknown" means
damage presence is unclear from the image. These are different: use "none" only when
the surface is clearly undamaged; use "unknown" when you cannot tell.

--- VALID IMAGE ---

valid_image is about whether the image is a usable, authentic photograph — NOT
about whether it shows the right damage or part.

- Set valid_image="false" ONLY when the image is technically unusable for automated
  review: it appears digitally manipulated or non-original (watermark, stock photo
  site, screenshot from the web, clearly edited), is completely unreadable, or is
  not a real photograph.

- Set valid_image="true" even when the image shows the wrong angle, wrong part, or
  contradicts the claim. Those issues belong in claim_status and risk_flags —
  not in valid_image.

--- RISK FLAGS: manual_review_required ---

Include "manual_review_required" ONLY when claim_status is "contradicted" AND at
least one of the following is also present: non_original_image, possible_manipulation,
or claim_mismatch. Do not add it based on user history alone or for any other reason.
"""


def format_requirements(reqs: list[dict]) -> str:
    lines = []
    for r in reqs:
        lines.append(f"- [{r['requirement_id']}] {r['applies_to']}: {r['minimum_image_evidence']}")
    return "\n".join(lines) if lines else "None"


def format_history(history: dict | None) -> str:
    if not history:
        return "No history available."
    return (
        f"Past claims: {history.get('past_claim_count', '?')} total "
        f"({history.get('accept_claim', '?')} accepted, "
        f"{history.get('rejected_claim', '?')} rejected, "
        f"{history.get('manual_review_claim', '?')} manual review). "
        f"Last 90 days: {history.get('last_90_days_claim_count', '?')}. "
        f"Flags: {history.get('history_flags', 'none')}. "
        f"Summary: {history.get('history_summary', '')}"
    )


def build_messages(
    claim: dict,
    image_paths: list[Path],
    history: dict | None,
    reqs: list[dict],
) -> list[dict]:
    claim_object = claim["claim_object"]
    ids = [image_id(p) for p in image_paths]
    part_options = ", ".join(OBJECT_PARTS.get(claim_object, ("unknown",)))

    user_text = f"""\
## Claim Context
Object type: {claim_object}
Image IDs (in order): {", ".join(ids)}

Conversation transcript:
{claim["user_claim"]}

## User History
{format_history(history)}

## Evidence Requirements
{format_requirements(reqs)}

## Output Schema
Return exactly this JSON object. Use only the allowed values listed.

{{
  "valid_image": "true" | "false",
  "evidence_standard_met": "true" | "false",
  "evidence_standard_met_reason": "<concise reason>",
  "risk_flags": "<flag1;flag2>" | "none",
  "issue_type": "<one of: {", ".join(ISSUE_TYPES)}>",
  "object_part": "<one of: {part_options}>",
  "claim_status": "supported" | "contradicted" | "not_enough_information",
  "claim_status_justification": "<concise, image-grounded explanation>",
  "supporting_image_ids": "<img_1;img_2>" | "none",
  "severity": "<one of: {", ".join(SEVERITIES)}>"
}}

Allowed risk_flags (semicolon-separated, or "none"):
{", ".join(RISK_FLAGS)}
"""

    content: list[dict] = [{"type": "text", "text": user_text}]

    for path in image_paths:
        if not path.exists():
            continue
        result = preprocess_image(path)
        if result is None:
            continue  # unreadable — skip, same as missing file
        data, mime = result
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def sanitize_single(value: str, field: str, allowed: tuple) -> str:
    """
    For fields that must hold a single allowed value, guard against the model
    returning a semicolon-separated list.  Pick the first part that appears in
    the allowed set; fall back to 'unknown' if none match.
    risk_flags is intentionally excluded — it is the only multi-value field.
    """
    parts = [p.strip() for p in value.split(";") if p.strip()]
    if len(parts) <= 1:
        return value
    for p in parts:
        if p in allowed:
            print(f"\n  [WARN] {field}: multi-value '{value}' → using '{p}'")
            return p
    fallback = "unknown" if "unknown" in allowed else parts[0]
    print(f"\n  [WARN] {field}: multi-value '{value}', no valid value → '{fallback}'")
    return fallback


# ---------------------------------------------------------------------------
# VLM call with retry
# ---------------------------------------------------------------------------

class RateLimitExhausted(Exception):
    """Raised by call_vlm when every retry was rejected due to rate limiting."""

def parse_vlm_json(text: str) -> dict:
    """Extract a JSON object from the model response."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Strip markdown fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: grab the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response:\n{text[:300]}")


def vlm_error_result() -> dict:
    return {
        "valid_image": "false",
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Automated analysis failed",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "VLM call failed; manual review required",
        "supporting_image_ids": "none",
        "severity": "unknown",
    }


def _smart_wait(exc) -> float:
    """
    Extract X-RateLimit-Reset from a 429 response body and return seconds to sleep.
    Falls back to None if the header is absent or the wait exceeds 120 s (e.g. daily limit).
    """
    try:
        body = exc.response.json()
        reset_ms = (
            body.get("error", {})
                .get("metadata", {})
                .get("headers", {})
                .get("X-RateLimit-Reset")
        )
        if reset_ms:
            wait = int(reset_ms) / 1000 - time.time()
            if 0 < wait <= 120:
                return wait + 0.5   # small buffer
    except Exception:
        pass
    return None


def _is_rate_limit(exc) -> bool:
    return getattr(exc, "status_code", None) == 429


def call_vlm(
    client,
    model: str,
    messages: list[dict],
    max_retries: int = 3,
    pre_call_delay: float = 0.0,
) -> dict:
    """
    Returns a result dict on success.
    Raises RateLimitExhausted if every retry was a 429 (lets callers try a fallback provider).
    Returns vlm_error_result() for any other terminal failure.
    """
    last_exc = None
    all_rate_limited = True
    for attempt in range(max_retries):
        if pre_call_delay > 0:
            time.sleep(pre_call_delay)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
            return parse_vlm_json(response.choices[0].message.content)
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit(exc):
                all_rate_limited = False
            if attempt < max_retries - 1:
                wait = _smart_wait(exc) if _is_rate_limit(exc) else None
                if wait is None:
                    wait = 2 ** attempt
                print(f"\n  [RETRY {attempt + 1}] {exc} — retrying in {wait:.1f}s")
                time.sleep(wait)

    if all_rate_limited:
        raise RateLimitExhausted(str(last_exc))
    print(f"\n  [ERROR] VLM call failed after {max_retries} attempts: {last_exc}")
    return vlm_error_result()


# ---------------------------------------------------------------------------
# Incremental output helpers
# ---------------------------------------------------------------------------

# Justifications that mark a row as a fallback/error (not a real VLM result).
# Rows with these values are eligible to be re-processed on resume.
_FALLBACK_JUSTIFICATIONS = {
    "VLM call failed; manual review required",
    "No images were available for automated review",
    "",
}


def load_done(path: Path) -> dict[tuple, dict]:
    """
    Read an existing output file and return a dict of already-successful rows
    keyed by (user_id, image_paths).  Fallback/error rows are excluded so they
    get re-processed on the next run.
    """
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("claim_status_justification") not in _FALLBACK_JUSTIFICATIONS:
                    done[(row["user_id"], row["image_paths"])] = row
        if done:
            print(f"  Resuming: {len(done)} already-successful row(s) loaded from {path.name}")
    except Exception as exc:
        print(f"  [WARN] Could not read existing output {path}: {exc}")
    return done


# ---------------------------------------------------------------------------
# Claim processing
# ---------------------------------------------------------------------------

def no_images_result() -> dict:
    return {
        "valid_image": "false",
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "No valid image files found on disk",
        "risk_flags": "damage_not_visible;manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "No images were available for automated review",
        "supporting_image_ids": "none",
        "severity": "unknown",
    }


def process_claim(
    claim: dict,
    user_history: dict[str, dict],
    requirements: list[dict],
    primary: tuple,
    fallback: tuple | None = None,
) -> dict:
    """
    primary / fallback are (client, model, pre_call_delay) tuples.
    On RateLimitExhausted from primary, retries with fallback if provided.
    """
    user_id = claim["user_id"]
    claim_object = claim["claim_object"]
    image_paths = resolve_images(claim["image_paths"])

    valid_paths = [p for p in image_paths if p.exists()]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"\n  [WARN] {user_id}: missing files: {[str(m) for m in missing]}")

    if not valid_paths:
        print(f"\n  [SKIP] {user_id}: no valid images — skipping VLM call")
        vlm = no_images_result()
    else:
        history = user_history.get(user_id)
        reqs = applicable_requirements(requirements, claim_object)
        messages = build_messages(claim, valid_paths, history, reqs)

        p_client, p_model, p_delay = primary
        try:
            vlm = call_vlm(p_client, p_model, messages, pre_call_delay=p_delay)
        except RateLimitExhausted:
            if fallback:
                f_client, f_model, f_delay = fallback
                print(f"\n  [FALLBACK] Groq rate-limited → OpenRouter")
                try:
                    vlm = call_vlm(f_client, f_model, messages, pre_call_delay=f_delay)
                except RateLimitExhausted:
                    print(f"\n  [ERROR] Both providers rate-limited")
                    vlm = vlm_error_result()
            else:
                print(f"\n  [ERROR] Rate limit exhausted, no fallback configured")
                vlm = vlm_error_result()

    part_options = OBJECT_PARTS.get(claim_object, ("unknown",))
    return {
        "user_id": user_id,
        "image_paths": claim["image_paths"],
        "user_claim": claim["user_claim"],
        "claim_object": claim_object,
        "evidence_standard_met": vlm.get("evidence_standard_met", "false"),
        "evidence_standard_met_reason": vlm.get("evidence_standard_met_reason", ""),
        "risk_flags": vlm.get("risk_flags", "none"),
        "issue_type": sanitize_single(
            vlm.get("issue_type", "unknown"), "issue_type", ISSUE_TYPES
        ),
        "object_part": sanitize_single(
            vlm.get("object_part", "unknown"), "object_part", part_options
        ),
        "claim_status": sanitize_single(
            vlm.get("claim_status", "not_enough_information"),
            "claim_status",
            ("supported", "contradicted", "not_enough_information"),
        ),
        "claim_status_justification": vlm.get("claim_status_justification", ""),
        "supporting_image_ids": vlm.get("supporting_image_ids", "none"),
        "valid_image": vlm.get("valid_image", "false"),
        "severity": sanitize_single(
            vlm.get("severity", "unknown"), "severity", SEVERITIES
        ),
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = REPO_ROOT / f"output_run_{ts}.csv"

    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review pipeline")
    parser.add_argument("--claims", type=Path, default=CLAIMS_CSV)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument(
        "--strategy",
        choices=["A", "B", "a", "b", "auto", "AUTO"],
        default="auto",
        help=(
            "A = Groq only, B = OpenRouter only, "
            "auto = Groq primary with per-claim OpenRouter fallback on rate limit"
        ),
    )
    args = parser.parse_args()

    load_dotenv()

    print("Loading data...")
    claims = load_claims(args.claims)
    user_history = load_user_history(USER_HISTORY_CSV)
    requirements = load_evidence_requirements(EVIDENCE_REQ_CSV)
    print(
        f"  claims={len(claims)}  "
        f"users_in_history={len(user_history)}  "
        f"evidence_rules={len(requirements)}"
    )

    strategy = args.strategy.upper()
    if strategy == "AUTO":
        groq_client, groq_model = get_vlm_client("A")
        or_client, or_model = get_vlm_client("B")
        primary  = (groq_client,  groq_model,  0.0)
        fallback = (or_client,    or_model,    4.0)
        print(f"  strategy=auto  primary={groq_model}  fallback={or_model}")
    elif strategy == "A":
        client, model = get_vlm_client("A")
        primary, fallback = (client, model, 0.0), None
        print(f"  strategy=A  model={model}")
    else:  # B
        client, model = get_vlm_client("B")
        primary, fallback = (client, model, 4.0), None
        print(f"  strategy=B  model={model}")

    done = load_done(args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    skipped = processed = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        print(f"\nProcessing {len(claims)} claims...")
        for i, claim in enumerate(claims, 1):
            key = (claim["user_id"], claim["image_paths"])
            label = f"[{i}/{len(claims)}] {claim['user_id']} / {claim['claim_object']}"

            if key in done:
                print(f"{label} → SKIP (already done)")
                writer.writerow(done[key])
                f.flush()
                skipped += 1
                continue

            print(label, end=" ", flush=True)
            result = process_claim(claim, user_history, requirements, primary, fallback)
            print(f"→ {result['claim_status']} | {result['issue_type']} | {result['severity']}")
            writer.writerow(result)
            f.flush()
            processed += 1

    print(f"\nWrote {skipped + processed} rows → {args.out}  (skipped={skipped}, processed={processed})")


if __name__ == "__main__":
    main()
