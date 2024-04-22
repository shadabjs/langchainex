import boto3
import os
import uuid
import time
import json
from jinja2 import Template

# Import necessary packages
from IPython.display import Audio

# Set up AWS clients
s3_client = boto3.client('s3', region_name='us-west-2')
transcribe_client = boto3.client('transcribe', region_name='us-west-2')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

# Specify the file and bucket names
file_name = 'dialog2.mp3'
bucket_name = os.environ['BucketName']

# Upload the audio file to S3
s3_client.upload_file(file_name, bucket_name, file_name)

# Start transcription job
job_name = 'transcription-job-' + str(uuid.uuid4())
response = transcribe_client.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': f's3://{bucket_name}/{file_name}'},
    MediaFormat='mp3',
    LanguageCode='en-US',
    OutputBucketName=bucket_name,
    Settings={
        'ShowSpeakerLabels': True,
        'MaxSpeakerLabels': 2
    }
)

# Wait for transcription job to complete
while True:
    status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)

# Check transcription job status
if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
    # Load transcript from S3
    transcript_key = f"{job_name}.json"
    transcript_obj = s3_client.get_object(Bucket=bucket_name, Key=transcript_key)
    transcript_text = transcript_obj['Body'].read().decode('utf-8')
    transcript_json = json.loads(transcript_text)
    
    # Extract conversation transcript
    transcript_items = transcript_json['results']['items']
    conversation = ""
    for item in transcript_items:
        if item['type'] == 'pronunciation':
            conversation += f"{item['alternatives'][0]['content']} "
    
    # Write conversation transcript to a text file
    with open('conversation.txt', 'w') as f:
        f.write(conversation)

# Define template for LLM prompt
template = Template("""
I need to summarize a conversation. The transcript of the conversation is as follows:

{{ conversation }}

The summary must contain a one-word sentiment analysis and a list of issues, problems, or causes of friction during the conversation. The output must be provided in the following JSON format:

{
    "sentiment": <sentiment>,
    "issues": [
        {
            "topic": <topic>,
            "summary": <issue_summary>
        }
    ]
}

Write the JSON output and nothing more.
""")

# Render template with conversation transcript
prompt = template.render(conversation=conversation)

# Write prompt to file
with open('prompt_template.txt', 'w') as f:
    f.write(prompt)

# Define model parameters
model_id = "amazon.titan-text-express-v1"
content_type = "application/json"
accept_type = "*/*"
prompt_text = prompt
max_token_count = 512
temperature = 0
top_p = 0.9

# Invoke LLM model
response = bedrock_runtime.invoke_model(
    modelId=model_id,
    contentType=content_type,
    accept=accept_type,
    body=json.dumps({
        "inputText": prompt_text,
        "textGenerationConfig": {
            "maxTokenCount": max_token_count,
            "temperature": temperature,
            "topP": top_p
        }
    })
)

# Parse LLM response
response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']

# Print LLM-generated summary
print(generation)

# Define the JSON output
summary_json = {
    "sentiment": "Positive",
    "issues": [
        {
            "topic": "Topic 1",
            "summary": "Summary of issue 1"
        },
        {
            "topic": "Topic 2",
            "summary": "Summary of issue 2"
        },
        # Add more issues as needed
    ]
}

# Write JSON output to file
with open('summary.json', 'w') as f:
    json.dump(summary_json, f)

