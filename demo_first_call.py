from http import client
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.2",
    input="In 2 sentences, define an applied AI for an application"
)

print(response)
