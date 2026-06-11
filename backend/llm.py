import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model based on environment variables.

    Uses LLM_PROVIDER and LLM_MODEL to determine the chat model.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            timeout=60,
            stop=[],
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. Choose from: "
            "anthropic, google, openai"
        )
