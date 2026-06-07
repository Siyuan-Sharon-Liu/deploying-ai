from openai import OpenAI
import os

USE_GATEWAY = (os.getenv('USE_GATEWAY', 'FALSE').upper() == 'TRUE')

def get_client(use_gateway: bool = USE_GATEWAY) -> OpenAI:
    if use_gateway:
        client = OpenAI(base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
                    api_key='any value',
                    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')})
    else:
        client = OpenAI()
    return client