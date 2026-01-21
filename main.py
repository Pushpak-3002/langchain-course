from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


def main():
    print("Hello from langchain-course!")

    information = """
    Jen-Hsun Huang, commonly anglicized as Jensen Huang, is a Taiwanese and American business executive,
    electrical engineer, and philanthropist who is the founder, president, and chief executive officer (CEO)
    of Nvidia...
    """

    summary_template = """
    Summarize the following information {information} about a person:
    1. A short summary of who they are.
    2. Their major accomplishments.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )

    #chain_openai = summary_prompt_template | llm_openai
    chain_gemini = summary_prompt_template | llm_gemini

    #response_openai = chain_openai.invoke({"information": information})
    response_gemini = chain_gemini.invoke({"information": information})

    print("\nResponse from OpenAI GPT-4:")
    #print(response_openai.content)

    print("\nResponse from Google Gemini 1.5 Pro:")
    print(response_gemini.content[0]["text"])


if __name__ == "__main__":
    main()
