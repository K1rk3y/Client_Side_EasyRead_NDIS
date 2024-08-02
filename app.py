import pandas as pd
import numpy as np
from ast import literal_eval
import pandas as pd
from scipy.spatial.distance import cosine
import tiktoken
from openai import OpenAI
import time
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('API_KEY')

# Function to check the job status and events
def check_job_status_and_events(job_id, client, delay=5):
    while True:
        # Retrieve the state of the fine-tune job
        response = client.fine_tuning.jobs.retrieve(job_id)
        print("Job ID:", response.id)
        print("Status:", response.status)
        print("Trained Tokens:", response.trained_tokens)
        
        # Retrieve the events of the fine-tune job
        events_response = client.fine_tuning.jobs.list_events(job_id)
        events = events_response.data
        events.reverse()
        
        for event in events:
            print(event.message)
        
        # Break the loop if the job is completed
        if response.status in ['succeeded', 'failed', 'cancelled']:
            print("Exited")
            return response
        
        # Wait for the specified delay before the next check
        time.sleep(delay)


def main(api_key):
    # Load OpenAI client
    client = OpenAI(api_key=api_key)

    response = client.files.create(
      file=open("training_data.jsonl", "rb"),
      purpose="fine-tune"
    )

    # Extract the file ID from the response
    training_file_id = response.id
    print(f"Uploaded file ID: {training_file_id}")

    # Create a fine-tuning job
    response = client.fine_tuning.jobs.create(
      training_file=training_file_id, 
      model="gpt-3.5-turbo",
      hyperparameters={
          "n_epochs":1
      }
    )

    job_id = response.id
    print(f"Fine tunning ID: {job_id}")

    # Call the function to monitor the job status and events
    response = check_job_status_and_events(job_id, client)

    fine_tuned_model_id = response.fine_tuned_model
    print("Fine-tuned model ID:", fine_tuned_model_id)

    for model in client.models.list():
        if model.owned_by != 'system':
            print(model)

main(api_key)