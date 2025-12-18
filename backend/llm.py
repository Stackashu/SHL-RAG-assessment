import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def generate_response(query: str, context: list[dict]) -> str:
    llm = get_llm()

    context_str = "\n".join(
        f"- {c['name']}: {c.get('test_type')}"
        for c in context
    )

    prompt = f"""
    User query: {query}

    Recommended assessments:
    {context_str}

    Explain why these assessments are suitable.
    """

    return llm.invoke(prompt).content
