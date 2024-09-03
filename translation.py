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
import json


load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(api_key=api_key)

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


def conversion(df, model="gpt-3.5-turbo", input="", condition_prompt='', max_len=1800, size="ada", debug=False, max_tokens=200, stop_sequence=None):
    context = create_context(input, df, max_len=max_len, size=size)
    
    with open('prompts.jsonl', 'r') as file:
        data = json.load(file)

    # Create the messages list
    system_prompt=f"You are a translator, your role is to translate the input text into easy read format based on BOTH the user input and the context below\nContext: {context}\nAdditional instruction: {condition_prompt}"
    messages = [{"role": "system", "content": system_prompt}]

    for conversation in data["conversations"]:
        messages.append({"role": "user", "content": conversation["user"]})
        messages.append({"role": "assistant", "content": conversation["assistant"]})

    messages.append({"role": "user", "content": input})

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = client.chat.completions.create(
            messages=messages,
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
    

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def create_df():
    # Create a list to store the text files
    texts = []

    # Get all the text files in the text directory
    for file in os.listdir("text/"):

        # Open the file and read the text
        with open("text/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)

    if not os.path.exists("processed"):
        os.mkdir("processed")

    df.to_csv('processed/scraped.csv')


def Wrapper(input, condition_prompt, model_id=None):
    model = None
    if model_id == None:
        model = fine_tune('training_data.jsonl', 'validation_data.jsonl')
    else:
        model = model_id

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Read data from the CSV file into a DataFrame
    create_df()
    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

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
            shortened += split_into_many(tokenizer, row[1]['text'], max_tokens)

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

    return conversion(df, model, input, condition_prompt, debug=False)

print("OPT: ", Wrapper("""Many people with an intellectual disability rely on the Disability Support Pension (DSP). The rules
about working and the DSP can discourage people with an intellectual disability from getting a job,
and from taking on more hours.
Often, people with an intellectual disability are only offered work in a sheltered workshop, also
known as an Australian Disability Enterprise (ADE), earning very low wages. The ADE deals with the
complexity of Centrelink, making it simpler to work there than in open employment.
When people with an intellectual disability earn money from their jobs, they lose some of money
they get from the DSP. This can be as high as 68c for every $1 they earn, which is a big penalty for
working. Inclusion Australia thinks that should change, and we want people with an intellectual
disability to be able to keep more of the money they earn at work.
People with an intellectual disability who don’t do what a Disability Employment Service (DES) tells
them to can have their DSP reduced or stopped. We don’t think this is a good idea, and research says
that this approach doesn’t help people get a job.""", '', 'ft:gpt-3.5-turbo-0125:intelife-group::A3CSPadd'))