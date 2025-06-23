from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.lambdalabs.com/v1")


def refine_translation(current_text: str, model: str = "deepseek-r1-671b"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in low-level concept identification and extraction, you extract the underlying low-level action or item from a piece of text provided by the user. The following are examples of such extractions:",
                },
                {
                    "role": "user",
                    "content": "An advocate is someone you can trust and who is on your side. Sometimes this is a family member, friend or a support person you know well.",
                },
                {
                    "role": "user",
                    "content": """ Extracted concept: "A supportive person" """,
                },
                {
                    "role": "user",
                    "content": "You give your permission for Intelife to collect information and talk toother people about your needs.",
                },
                {
                    "role": "user",
                    "content": """ Extracted concept: "Discuss with people" """,
                },
                {
                    "role": "user",
                    "content": "You will be treated fairly and helped to feel comfortable. If you need to speak in your own language, we can get an interpreter to help",
                },
                {"role": "user", "content": """ Extracted concept: "Get help" """},
                {
                    "role": "system",
                    "content": "Now extract the concept from the following text.",
                },
                {"role": "user", "content": current_text},
            ],
            temperature=0,
            max_tokens=2000,
        )

        opt = response.choices[0].message.content

        return opt

    except Exception as e:
        print(f"Error in refinement: {str(e)}")
        return current_text, {"error": str(e)}


def extract_last_quoted(text: str) -> str | None:
    matches = re.findall(r'"([^"]*)"', text)
    return matches[-1] if matches else None


def translate(input_text):
    results = refine_translation(current_text=input_text)
    print("RESULTS: ", results)
    opt = extract_last_quoted(results)

    return opt
