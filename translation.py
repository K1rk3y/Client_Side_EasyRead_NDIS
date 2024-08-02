import pandas as pd
import numpy as np
from ast import literal_eval
import pandas as pd
from scipy.spatial.distance import cosine
import tiktoken
from openai import OpenAI
import os
from dotenv import load_dotenv
from fine_tune import fine_tune


load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(api_key=api_key)

# Function to remove newlines from a Series
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(tokenizer, text, max_tokens):
    sentences = text.split('. ')
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


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Define functions for responses
def create_context(input, df, max_len=1800, size="ada"):
    q_embeddings = get_embedding(input)
    df["distances"] = df["embeddings"].apply(lambda x: cosine(q_embeddings, x))
    returns = []
    cur_len = 0
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])
    return "\n\n###\n\n".join(returns)


def conversion(df, model="gpt-3.5-turbo", condition_prompt='', max_len=1800, size="ada", debug=False, max_tokens=300, stop_sequence=None):
    context = create_context(input, df, max_len=max_len, size=size)
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        prompt=f"You are a translator, your role is to translate the input text into easy read format based on BOTH the user input and the context below\nContext: {context}\nAdditional infomation: {condition_prompt}"
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": """Inclusion Australia wants big changes to the current Disability Employment Services (DES) because they are failing people with an intellectual disability and their families. 
                                                Currently, only 3.1 percent of those using DES are people with an intellectual disability. 
                                                People with an intellectual disability have the right to work, and to be paid fairly for that work.
                                                Most people with an intellectual disability do not have access to paid work in open and self-
                                                employment. This means that most people with an intellectual disability live in poverty,
                                                with no opportunity to have the same lives as non-disabled people, and other people with
                                                disability. People with an intellectual disability have to find their way through lots of different systems when
                                                they are looking for support to get and keep a job. These systems are Disability Employment Services
                                                (DES), the National Disability Insurance Scheme (NDIS), the Disability Support Pension (DSP),
                                                Centrelink and school. We think all this needs to be much easier. """},
                {"role": "assistant", "content": """Inclusion Australia made a report about Disability Employment Services. 
                                                    DES for short. 
                                                    DES providers are services that help people with disability find jobs."""},
                {"role": "system", "content": prompt},
                {"role": "user", "content": """There has been lots of research into different ways to support people with an intellectual disability
                                                at work. That research has found there are ways that are good and we've used that research in our
                                                submission. Inclusion Australia's Our Voice Committee, who are all people with an intellectual disability, say that
                                                “People with disabilities have the right to work in the open market like anyone else, and get the
                                                training and support they need; this means no more sheltered workshops."""},
                {"role": "assistant", "content": """We want DES to work better for people with intellectual disability.
                                                    Inclusion Australia will send this report to the government.
                                                    We want them to make big changes to DES.
                                                    This document will talk about some of the main ideas in the report."""},
                {"role": "system", "content": prompt},
                {"role": "user", "content": """Lots of DES don’t know about people with an intellectual disability, or about what works to get a job
                                                and keep people at work. People with an intellectual disability and their families find it hard to get accessible, independent
                                                information about employment, as well as the NDIS and Centrelink. A Centre of Excellence, as well as specialist DES will know what kinds of programs and supports work
                                                the best for people with an intellectual disability and their families. They will make information accessible and available to people with an intellectual disability and their
                                                families so they can find their way through complicated systems, like the NDIS and Centrelink."""},
                {"role": "assistant", "content": """We found out that lots of people with intellectual disability do not get the same
                                                    money and jobs as other people.
                                                    People with intellectual disability have the right to
                                                    • Work just like everyone else
                                                    • Be paid well"""},
                {"role": "system", "content": prompt},
                {"role": "user", "content": """There are lots of obstacles in the way of people with an intellectual disability who want to get and
                                                keep a job. These obstacles can be the low expectations that other people have, and not having the same kinds of choices.
                                                Big systems, like the NDIS and DES, can also be an obstacle to working in a regular job or having a
                                                business. We want to change that."""}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model
        )

        return response.choices[0].message.content
    
    except Exception as e:
        print(e)
        return ""


def Wrapper(crawler, condition_prompt):
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Read data from the CSV file into a DataFrame
    df = pd.read_csv(crawler)

    # Check if any text column exists in the DataFrame
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    if len(text_columns) == 0:
        print("No text column found in the DataFrame. Exiting.")
        exit()

    # Concatenate all text columns into a single 'text' column
    df['text'] = df[text_columns].apply(lambda row: ' '.join(row.dropna()), axis=1)

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    # df['n_tokens'].hist()

    # Set max_tokens
    max_tokens = 500

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    # Create DataFrame from shortened texts
    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # df['n_tokens'].hist()

    df['embeddings'] = df.text.apply(lambda x: get_embedding(x))

    # Save embeddings to CSV
    df.to_csv('embeddings.csv')
    df.head()

    # Read embeddings from CSV
    df = pd.read_csv('embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    df.head()

    model = fine_tune('training_data.jsonl', 'validation_data.jsonl')

    return conversion(df, model, input=input, condition_prompt=condition_prompt, debug=False)

Wrapper('output.csv', '')