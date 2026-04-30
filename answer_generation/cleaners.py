"""Pure cleaners for Twiga answers.

regex_clean is deterministic and faithful — strips decoration only.
llm_clean uses DeepSeek to remove pedagogical fluff with model judgment.

No file I/O. CSV runners (clean_answers.py) call these in a loop.
"""

import os
import re

from openai import OpenAI

_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

LLM_CLEAN_MODEL = "deepseek-v4-pro"


# Conservative emoji ranges — covers the common pictograph/symbol/dingbat blocks
# without sweeping up non-emoji unicode (CJK, accents, etc.)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric ext.
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess / symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs ext.
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "‍"                  # zero-width joiner
    "️"                  # variation selector
    "]+",
    flags=re.UNICODE,
)


def regex_clean(raw: str) -> str:
    """Strip decoration only — preserves all factual text verbatim.

    Removes: emojis, markdown headers/bold/italic/horizontal rules,
    leading greetings, trailing 'Teaching Tip' and 'Would you like' CTAs.
    """
    text = raw

    # 1. Strip emojis
    text = EMOJI_PATTERN.sub("", text)

    # 2. Drop trailing 'Teaching Tip' section (common closer in the bot's output)
    text = re.sub(r"(?is)\n\s*\*?\s*Teaching Tip.*$", "", text)

    # 3. Drop trailing CTA ("Would you like ... ?" through end of doc)
    text = re.sub(r"(?is)\n\s*Would you like[^?]*\?.*$", "", text)

    # 4. Drop the greeting phrase ("Hello/Hi/Hey <Name>!") — first occurrence only,
    # since these greetings sometimes appear after a title line rather than at doc start.
    text = re.sub(
        r"\b(Hello|Hi|Hey|Greetings)\b[^!?\.\n]*!\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    # 5. Strip horizontal rule lines (---, ***)
    text = re.sub(r"(?m)^\s*[-*]{3,}\s*$", "", text)

    # 6. Strip markdown header prefixes (#, ##, ### at line start)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)

    # 7. Strip bold/italic wrappers, longest first to avoid leaving lone markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*([^\*\n]+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)

    # 8. Collapse 3+ blank lines down to 2
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def llm_clean(question: str, raw_answer: str) -> str:
    """LLM-judgment cleaner — strips pedagogical fluff that regex can't catch.

    NOTE: in our 71-row eval, this paraphrased ~95% of rows (dropped factual
    tokens that regex preserved). Prefer regex_clean for evaluation; keep this
    for comparison / spot-checking.
    """
    prompt = f"""
    You are extracting the factual answer from a verbose chatbot response so it can be used in RAG evaluation.

    Given a question and a wordy answer, return ONLY the substantive answer content. Remove:
    - Greetings (e.g. "Hello", "Hi there")
    - Teaching tips, suggestions, or pedagogical advice
    - Closing offers / CTAs (e.g. "Would you like me to...", "Let me know if...")
    - Source references / reading notes (e.g. "According to the textbook", "Based on Chapter X")
    - Emojis and decorative symbols
    - Markdown formatting (asterisks for bold/italic, hash headers, bullet markers, horizontal rules)

    Keep:
    - All factual content that directly answers the question
    - Definitions, characteristics, lists of facts (just write them as plain prose or comma-separated)
    - Numbers, ranges, units
    - The original wording — do NOT paraphrase, rewrite, or "improve" the text. Only remove the items listed above.

    Output ONLY the cleaned answer as plain text. No preamble, no quotes, no explanation.

    Question: {question}

    Raw Answer:
    {raw_answer}
    """

    response = _client.chat.completions.create(
        model=LLM_CLEAN_MODEL,
        max_tokens=16384,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        print(f"    finish_reason={choice.finish_reason!r} usage={response.usage!r}")
        raise ValueError("Empty response from LLM cleaner")
    return content.strip()
