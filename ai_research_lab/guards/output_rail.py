from guardrails import Guard
from guardrails.hub import ValidLength, IsValidMarkdown


output_guard = (
    Guard()
    .use(ValidLength, min=150, on_fail="exception")
    .use(IsValidMarkdown, on_fail="exception")
)
