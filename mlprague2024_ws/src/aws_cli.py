import os
import json
import boto3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

EMBEDDING_MODEL_CONFIG = {
    'modelId': 'amazon.titan-embed-text-v1',
    'accept': '*/*',
    'contentType': 'application/json'
}

LANGUAGE_MODEL_CONFIG = {
    'modelId': 'mistral.mistral-large-2402-v1:0',
    'accept': '*/*',
    'contentType': 'application/json'
}

class Client:
    def __init__(self):
        session = boto3.Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
        self.client = session.client(service_name='bedrock-runtime', region_name='us-west-2')

    def embed_text(self, text):
        body = json.dumps({"inputText": text})
    
        response = self.client.invoke_model(**EMBEDDING_MODEL_CONFIG, body=body)
        response_body = json.loads(response.get('body').read())
    
        embedding = response_body.get('embedding')
    
        return embedding

    def execute_prompt(self, prompt):
        body = json.dumps({
            'prompt': prompt,
            'max_tokens': 512,
            'top_p': 1.0,
            'temperature': 0.3,
        })
    
        response = self.client.invoke_model(**LANGUAGE_MODEL_CONFIG, body=body.encode('utf-8'))
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        output = response_body.get('outputs', [{}])[0] 
        generated_text = output.get('text').strip()
    
        return generated_text