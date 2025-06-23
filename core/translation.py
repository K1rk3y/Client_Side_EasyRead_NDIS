import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.spatial.distance import cosine
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.lambdalabs.com/v1")

# Initialize local models
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/text-embedding-ada-002")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def split_into_many(text, max_tokens):
    sentences = text.split(". ")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    if chunk:
        chunks.append(". ".join(chunk) + ".")
    return chunks


def get_embedding(text):
    text = text.replace("\n", " ")
    return embedding_model.encode(text).tolist()


def create_context(input, df, max_len=1800):
    q_embeddings = get_embedding(input)
    df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))

    returns = []
    cur_len = 0

    for i, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def get_refinement_criteria() -> str:
    return """Please evaluate the translation based on the following criteria:
    1. Is the easy-read translation clear and really simple to understand? 
    2. Have you replaced the use of multi-syllable words?
    3. Does it maintain the original meaning while being simpler?
    4. Is it structured in a way that's easy to follow?
    5. Is the simplification level consistent throughout?
    6. Although each sentence in the translation have the same background meaning, do they still have enough separation between them so that each sentence can repersent a meaningful sub-concept?
    7. Each sub-concept must only have one corresponding sentence and the sub-concept themselves must be important enough to be included. Is the number of sentences too many and can be reduced?
    8. Imagine you are a human reader and the target audience of this text, what are the important points in this text that is reasonably relevant to you? Can the sentences that does not contain these points be removed?
    9. Is the resulting translation in plain sentences that are period seprated and in one paragraph only (no dot points, no colon, etc)? 

    Then, provide an improved version of the translation that addresses the issues.
    Ensure that among the imporved outputs, there are no sentences with duplicating meaning of others.
    Ensure the improved version of the translation is wrapped in double quotes."""


def refine_translation(
    client: OpenAI,
    current_text: str,
    original_input: str,
    context: str,
    model: str = "deepseek-r1-671b",
) -> Tuple[str, Dict[str, Any]]:
    """Refine the given translation based on evaluation criteria."""
    refinement_prompt = f"""Original Text: {original_input}

    Current Translation:
    {current_text}

    Context from Similar Texts:
    {context}

    {get_refinement_criteria()}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in converting text to easy-read format while maintaining accuracy and clarity.",
                },
                {"role": "user", "content": refinement_prompt},
            ],
            temperature=0,
            max_tokens=2000,
        )

        feedback = response.choices[0].message.content
        improved_text = feedback.split("improved version")[-1].strip()

        return improved_text, {"full_feedback": feedback}

    except Exception as e:
        print(f"Error in refinement: {str(e)}")
        return current_text, {"error": str(e)}


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def create_df():
    texts = []
    for file in os.listdir("app/text/"):
        with open("app/text/" + file, "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append(
                (
                    file[11:-4]
                    .replace("-", " ")
                    .replace("_", " ")
                    .replace("#update", ""),
                    text,
                )
            )

    df = pd.DataFrame(texts, columns=["fname", "text"])
    df["text"] = df.fname + ". " + remove_newlines(df.text)

    if not os.path.exists("processed"):
        os.mkdir("processed")

    df.to_csv("processed/scraped.csv")
    return df


def prepare_embeddings_df():
    """Prepare and return DataFrame with embeddings."""
    if os.path.exists("embeddings.csv"):
        df = pd.read_csv("embeddings.csv", index_col=0)
        df["embeddings"] = df["embeddings"].apply(literal_eval).apply(np.array)
        return df

    # Create new embeddings if file doesn't exist
    df = create_df()
    df.columns = ["title", "text"]

    df["n_tokens"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
    max_tokens = 1000

    shortened = []
    for row in df.iterrows():
        if row[1]["text"] is None:
            continue
        if row[1]["n_tokens"] > max_tokens:
            shortened += split_into_many(row[1]["text"], max_tokens)
        else:
            shortened.append(row[1]["text"])

    df = pd.DataFrame(shortened, columns=["text"])
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df["embeddings"] = df.text.apply(get_embedding)

    df.to_csv("embeddings.csv")
    return df


def iterative_translation(
    input_text: str, n_iterations: int = 1, model: str = "deepseek-r1-671b"
) -> List[Tuple[str, Dict[str, Any]]]:
    # Prepare embeddings DataFrame
    df = (
        prepare_embeddings_df()
        if not os.path.exists("embeddings.csv")
        else pd.read_csv("embeddings.csv", index_col=0)
    )
    if "embeddings" in df.columns:
        df["embeddings"] = df["embeddings"].apply(literal_eval).apply(np.array)

    # Get context for the translation
    context = create_context(input_text, df)

    input_text = "User input: " + input_text + "\nContext: {context}"

    # Initial translation
    system_prompt = f"You are a translator, your role is to translate the user input text into easy read format based on BOTH the user input and the context."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.7,
            max_tokens=2000,
            model=model,
        )
        current_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error in initial translation: {str(e)}")
        return []

    results = [(current_text, {"stage": "initial"})]

    # Refinement loop
    for i in tqdm(range(n_iterations), desc="Refining translation"):
        try:
            refined_text, feedback = refine_translation(
                client=client,
                current_text=current_text,
                original_input=input_text,
                context=context,
                model=model,
            )

            results.append(
                (refined_text, {"stage": f"refinement_{i+1}", "feedback": feedback})
            )
            current_text = refined_text

        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            break

    return results


def save_refinement_history(
    results: List[Tuple[str, Dict[str, Any]]], filename: str = "translation_history.txt"
) -> Optional[str]:
    last_quoted_line = None

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Translation Refinement History ===\n\n")
        for i, (text, metadata) in enumerate(results):
            f.write(f"\n--- Stage: {metadata['stage']} ---\n")

            # Check each line in the text for quotes
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    last_quoted_line = line

            f.write(text + "\n")

            if i > 0 and "feedback" in metadata:
                f.write("\nFeedback from previous version:\n")
                feedback = metadata["feedback"]["full_feedback"]
                f.write(feedback + "\n")

                # Also check feedback for quoted lines
                for line in feedback.split("\n"):
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        last_quoted_line = line

            f.write("\n" + "=" * 50 + "\n")

    return last_quoted_line


def translate(input_text):
    # Run iterative translation
    results = iterative_translation(input_text, n_iterations=1)
    opt = save_refinement_history(results)
    print("TRANSLATION: ", opt)

    return opt
