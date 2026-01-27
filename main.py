from dotenv import load_dotenv
load_dotenv()
import os

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

@traceable(name="person_summary_pipeline")
def main():
    print("TRACING:", os.getenv("LANGCHAIN_TRACING_V2"))
    print("ENDPOINT:", os.getenv("LANGCHAIN_ENDPOINT"))
    print("PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
    print("API KEY PRESENT:", bool(os.getenv("LANGCHAIN_API_KEY")))
    
    
    print("\n \n ......Hello from langchain-course!...... \n \n")

    information = """
    Jen-Hsun Huang , commonly anglicized as Jensen Huang, is a Taiwanese and American business executive, electrical engineer, and philanthropist who is the founder, president, and chief executive officer (CEO) of Nvidia, the world's largest company by market capitalization. As of January 2026, Forbes estimates his net worth to be US$164.1 billion, making Huang the eighth-wealthiest individual in the world.[3]
    Born to Taiwanese American immigrants, Huang spent his childhood in Taiwan and Thailand before moving to the United States, where he was a student in Kentucky and Oregon. After earning his master's degree from Stanford University, Huang launched Nvidia in 1993 from a local Denny's restaurant at age 30 and has remained president and CEO since its founding. He led the company out of near-bankruptcy during the 1990s and oversaw its expansion into GPU production, high-performance computing, and artificial intelligence (AI).
    Under Huang, Nvidia experienced rapid growth during the AI boom, becoming the first company to reach a market capitalization of over $5 trillion in October 2025.[4] In 2021 and 2024, Time magazine included Huang in their Time 100 list of the most influential people. In 2025, he was named as one of the "Architects of AI" for Time's Person of the Year.
    """

    summary_template = """
    Summarize the following information {information} about a person:
    1. A short summary of who they are.
    2. Their major accomplishments.
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )

    llm_ollama = ChatOllama(
        model="gemma3:270m",
        temperature=0
    )

    chain_gemini = prompt | llm_gemini
    chain_ollama = prompt | llm_ollama

    response_gemini = chain_gemini.invoke({"information": information})
    response_ollama = chain_ollama.invoke({"information": information})

    print("\nGemini Response:\n", response_gemini.content)
    print("\nOllama Response:\n", response_ollama.content)

if __name__ == "__main__":
    main()