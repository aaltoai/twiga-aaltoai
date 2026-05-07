"""
Minimal LLM test without Twiga app dependencies.
Tests calling LLM directly via the configured LLM provider.
Supports: openai, together, ollama, google, modal
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment
load_dotenv()

async def test_llm_call():
    """Test calling LLM directly without Twiga dependencies"""
    
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL_NAME", "")
    
    question = "What is the capital of Tanzania?"
    
    if not llm_api_key:
        return {"success": False, "error": "LLM_API_KEY not set in .env"}
    
    try:
        if llm_provider == "openai":
            from openai import AsyncOpenAI
            
            if not llm_model:
                llm_model = "gpt-4o-mini"
            
            print(f"\n[OpenAI] Testing LLM with model: {llm_model}")
            print(f"Question: {question}")
            
            client = AsyncOpenAI(api_key=llm_api_key)
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=500,
            )
            answer = response.choices[0].message.content
            print(f"Answer: {answer}\n")
            return {"success": True, "provider": "OpenAI", "model": llm_model, "answer": answer}
            
        elif llm_provider == "together":
            from openai import AsyncOpenAI
            
            if not llm_model:
                llm_model = "meta-llama/Llama-3-70b-chat-hf"
            
            print(f"\n[Together] Testing LLM with model: {llm_model}")
            print(f"Question: {question}")
            
            # Together AI is OpenAI-compatible; use OpenAI SDK with Together's base URL
            client = AsyncOpenAI(
                api_key=llm_api_key,
                base_url="https://api.together.xyz/v1",
            )
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=500,
            )
            answer = response.choices[0].message.content
            print(f"Answer: {answer}\n")
            return {"success": True, "provider": "Together", "model": llm_model, "answer": answer}
            
        elif llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if not llm_model:
                llm_model = "gemini-2.5-flash"
            
            print(f"\n[Google Gemini] Testing LLM with model: {llm_model}")
            print(f"Question: {question}")
            
            client = ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=llm_api_key,
            )
            response = await client.ainvoke(question)
            answer = response.content
            print(f"Answer: {answer}\n")
            return {"success": True, "provider": "Google Gemini", "model": llm_model, "answer": answer}
            
        elif llm_provider == "ollama":
            from langchain_community.llms import Ollama
            
            if not llm_model:
                llm_model = "llama2"
            
            print(f"\n[Ollama] Testing LLM with model: {llm_model}")
            print(f"Question: {question}")
            print(f"Note: Make sure Ollama is running on http://localhost:11434")
            
            client = Ollama(model=llm_model)
            answer = await client.ainvoke(question)
            print(f"Answer: {answer}\n")
            return {"success": True, "provider": "Ollama", "model": llm_model, "answer": answer}
            
        else:
            return {"success": False, "error": f"Unsupported LLM provider: {llm_provider}. Supported: openai, together, google, ollama"}
            
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        print(f"Error calling LLM: {error_msg}")
        traceback.print_exc()
        return {"success": False, "provider": llm_provider, "error": error_msg}


if __name__ == "__main__":
    print("=" * 70)
    print("Direct LLM Test (No Twiga Stack)")
    print("=" * 70)
    
    provider = os.getenv("LLM_PROVIDER", "not set").lower()
    api_key_set = "Yes" if os.getenv("LLM_API_KEY") else "No"
    model = os.getenv("LLM_MODEL_NAME", "default")
    
    print(f"Configuration:")
    print(f"  LLM_PROVIDER: {provider}")
    print(f"  LLM_API_KEY set: {api_key_set}")
    print(f"  LLM_MODEL_NAME: {model}")
    print("=" * 70)
    
    result = asyncio.run(test_llm_call())
    print(f"\nResult: {json.dumps(result, indent=2)}")
