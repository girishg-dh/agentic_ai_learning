from guardrails import Guard
from guardrails.hub import ProfanityFree, PreventPromptInjection


PROMPT = """
Guardrails AI is a Python library that validates and corrects output from 
large language models.

Given the user's research topic, please validate it.

User Topic: ${topic}

"""

input_gaurd = (
    Guard()
    .use(ProfanityFree, on_fail="exception")
    .use(PreventPromptInjection, on_fail="exception")
)
