import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the PDF, say that the document does not provide enough information. "
    "Never invent a page number."
)


def _complete(user_content: str) -> str:
    """Send one grounded user message to the model under the shared system prompt."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def answer_from_pages(pages: list[dict], message: str) -> str:
    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    return _complete(f"PDF text:\n{document_text}\n\nmessage: {message}")


def answer_with_prompt(prompt: str) -> str:
    """Answer from an already-grounded prompt.

    The caller (rag.build_grounded_user_prompt) has already combined the recent
    conversation history, the retrieved evidence, and the current question.
    """
    return _complete(prompt)