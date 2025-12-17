import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        return None
    
    return ChatGoogleGenerativeAI(
        model="gemini-flash-lite", 
        google_api_key=api_key
    )

def generate_response(query: str, context: list[dict]) -> str:
    """
    Generates a response using LangChain and Google Gemini.
    """
    llm = get_llm()
    if not llm:
        return "I'm sorry, I cannot generate a response right now because the AI service is not configured."

    try:
        # Prepare context string
        context_str = "\n".join([
            f"- {item.get('name')}: {item.get('description')} (Type: {item.get('test_type')})"
            for item in context
        ])

        template = """
        You are a helpful assistant for SHL assessment recommendations.
        
        User Query: "{query}"
        
        Here are the most relevant assessments found in our catalog:
        {context_str}
        
        Based on the user's query and these recommendations, provide a concise and helpful summary. 
        Explain why these specific tests might be good for them. If the recommendations don't seem to match perfectly, acknowledge that but suggest the closest options.
        Do not invent new tests, only refer to the ones provided.
        """
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        
        result = chain.invoke({"query": query, "context_str": context_str})
        # print(result) 
        return result.content
        
    except Exception as e:
        print(f"Error {e}")
        return "Error occurede with LLM"
