import pandas as pd
import numpy as np
from ast import literal_eval
import pandas as pd
from scipy.spatial.distance import cosine
import tiktoken
from openai import OpenAI

# Load OpenAI client
client = OpenAI(api_key='sk-')

response = client.files.create(
  file=open("training_data.jsonl", "rb"),
  purpose="fine-tune"
)

# Extract the file ID from the response
training_file_id = response['id']
print(f"Uploaded file ID: {training_file_id}")

# Create a fine-tuning job
response = client.fine_tuning.jobs.create(
  training_file=training_file_id, 
  model="gpt-4o-mini"
)

job_id = response['id']
print(f"Fine tunning ID: {job_id}")

# List 10 fine-tuning jobs
client.fine_tuning.jobs.list(limit=10)

# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve(job_id)
