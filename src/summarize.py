"""Grounded LLM summarisation + automatic faithfulness check.

The LLM never sees raw data it could freely invent about: it receives a small
JSON "evidence pack" (counts, percentages, representative reviews with IDs)
and must cite review IDs like [R12]. verify_summary() then checks that every
citation exists and every larger number appears in the evidence. This is our
mitigation for hallucination.

Backends:
  template - deterministic, no model needed (always works; used as fallback)
  claude   - Anthropic API (set ANTHROPIC_API_KEY, optional SUMMARY_CLAUDE_MODEL)
  local    - open model through Hugging Face (default Qwen2.5-1.5B-Instruct)
"""
import json
import os
import re

SYSTEM_PROMPT = (
    "You are an analyst writing a short briefing for a hotel/restaurant manager. "
    "Use ONLY the evidence JSON provided. Never invent numbers, reviews or causes. "
    "Write 4-6 sentences: the 3 biggest problems first (with counts or percentages "
    "taken from the evidence), then one positive strength, then one concrete action. "
    "After each claim cite supporting review IDs in square brackets, e.g. [R12]. "
    "Reply in {language}."
)


def build_evidence(aspect_summary_df, clusters, total_reviews):
    """Compact, citable evidence pack for the LLM."""
    return {
        "total_reviews": int(total_reviews),
        "aspects": aspect_summary_df.to_dict("records"),
        "complaint_clusters": clusters,
    }


def make_prompt(evidence, language="English"):
    return (SYSTEM_PROMPT.format(language=language) +
            "\n\nEVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False, indent=1))


# --------------------------- backends ---------------------------------
class TemplateBackend:
    name = "template"

    def generate(self, evidence, language="English"):
        parts = []
        total = evidence["total_reviews"]
        for c in evidence["complaint_clusters"][:3]:
            ex = ", ".join(f"[{e['id']}]" for e in c["examples"][:2])
            terms = ", ".join(c["terms"][:3]) or c["top_aspect"]
            parts.append(f"{terms} ({c['size']} complaints, {c['share_pct']}% of "
                         f"negative comments, mostly about {c['top_aspect']}) {ex}")
        pos = sorted(evidence["aspects"], key=lambda a: a["positive"], reverse=True)
        strength = (f" Strongest aspect: {pos[0]['aspect']} "
                    f"({pos[0]['positive']} positive mentions)." if pos else "")
        if not parts:
            return f"Across {total} reviews no recurring complaints were found." + strength
        return (f"Across {total} reviews, the main complaints are: "
                + "; ".join(parts) + "." + strength)


class ClaudeBackend:
    name = "claude"

    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("SUMMARY_CLAUDE_MODEL", "claude-sonnet-5-5")

    def generate(self, evidence, language="English"):
        msg = self.client.messages.create(
            model=self.model, max_tokens=700, temperature=0.2,
            messages=[{"role": "user", "content": make_prompt(evidence, language)}])
        return msg.content[0].text.strip()


class LocalHFBackend:
    name = "local"

    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model, device_map="auto")

    def generate(self, evidence, language="English"):
        messages = [{"role": "user", "content": make_prompt(evidence, language)}]
        out = self.pipe(messages, max_new_tokens=400, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()


def get_backend(name="auto"):
    """auto = Claude if an API key exists, otherwise the template backend."""
    if name == "auto":
        name = "claude" if os.getenv("ANTHROPIC_API_KEY") else "template"
    try:
        if name == "claude":
            return ClaudeBackend()
        if name == "local":
            return LocalHFBackend()
    except Exception as e:  # missing package / key / model download failed
        print(f"[summarize] backend '{name}' unavailable ({e}); using template")
    return TemplateBackend()


# ----------------------- faithfulness check ----------------------------
def verify_summary(summary, evidence):
    """Check citations and numbers in the summary against the evidence.

    Returns a dict with the share of valid citations and supported numbers.
    Numbers <= 10 without % are ignored (ordinals like '3 problems').
    """
    allowed_ids = {e["id"] for c in evidence["complaint_clusters"] for e in c["examples"]}
    cited = re.findall(r"\bR\d+\b", summary)
    invalid = sorted({c for c in cited if c not in allowed_ids})

    allowed_nums = {evidence["total_reviews"]}
    for a in evidence["aspects"]:
        for k in ("mentions", "negative", "neutral", "positive", "negative_pct"):
            allowed_nums.add(float(a[k]))
    for c in evidence["complaint_clusters"]:
        allowed_nums.update({float(c["size"]), float(c["share_pct"]),
                             round(c["share_pct"]), float(round(c["share_pct"]))})
    allowed_nums = {float(x) for x in allowed_nums}
    allowed_nums |= {round(x) for x in allowed_nums}

    text = re.sub(r"\bR\d+\b", " ", summary)
    unsupported, checked = [], 0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%?)", text):
        val, pct = float(m.group(1)), bool(m.group(2))
        if val <= 10 and not pct:
            continue
        checked += 1
        if val not in allowed_nums and round(val) not in allowed_nums:
            unsupported.append(m.group(0).strip())

    n_checks = len(cited) + checked
    ok = (len(cited) - len(invalid)) + (checked - len(unsupported))
    return {
        "citations": len(cited), "invalid_citations": invalid,
        "numbers_checked": checked, "unsupported_numbers": unsupported,
        "faithfulness": round(ok / n_checks, 3) if n_checks else 1.0,
        "passed": not invalid and not unsupported,
    }
