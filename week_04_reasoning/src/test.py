import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('/Users/girish.gupta/work/learn/learning_reasources/agentic_ai_learning/.env')



client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "test-123", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "Lerner", # Optional. Site title for rankings on openrouter.ai.
  },
  model="openai/gpt-oss-20b:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
print(completion.choices[0].message.content)