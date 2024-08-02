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
from validation import validator


load_dotenv()
api_key = os.getenv('API_KEY')
client = OpenAI(api_key=api_key)


def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id

# Function to check the job status and events
def check_job_status_and_events(job_id, client, delay=5):
    while True:
        # Retrieve the state of the fine-tune job
        response = client.fine_tuning.jobs.retrieve(job_id)
        print("Job ID: ", response.id)
        print("Status: ", response.status)
        print("Trained Tokens: ", response.trained_tokens)
        
        # Retrieve the events of the fine-tune job
        events_response = client.fine_tuning.jobs.list_events(job_id)
        events = events_response.data
        events.reverse()
        print("Current event: ", events[-1])
        
        # Break the loop if the job is completed
        if response.status in ['succeeded', 'failed', 'cancelled']:
            print(f"Exited with status: {response.status}")
            return response
        
        # Wait for the specified delay before the next check
        time.sleep(delay)


def main(training_file_name, validation_file_name):
    if not validator(training_file_name):
        return
    
    training_file_id = upload_file(training_file_name, "fine-tune")
    validation_file_id = upload_file(validation_file_name, "fine-tune")

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)

    # Create a fine-tuning job
    response = client.fine_tuning.jobs.create(
      training_file=training_file_id, 
      validation_file=validation_file_id,
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


main('training_data.jsonl', 'validation_data.jsonl')