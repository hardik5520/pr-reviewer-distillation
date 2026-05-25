
# To run uv run python scripts/test_live.py
"""Send a real PR review request to the live vLLM endpoint."""
from openai import OpenAI

client = OpenAI(
    base_url="https://hardik5520--pr-reviewer-vllm-serve.modal.run/v1",
    api_key="not-needed",  # vLLM doesn't require auth for our setup
)

diff = """diff --git a/auth.py b/auth.py
@@ -10,7 +10,8 @@ def authenticate(token):
-    user = db.query(User).filter_by(token=token).first()
+    user = db.query(User).filter_by(token=token).first()
+    user.last_login = datetime.now()
     return user
"""

response = client.chat.completions.create(
    model="pr-reviewer-7b-instruct-awq",
    messages=[
        {
            "role": "system",
            "content": "You are an expert code reviewer. Follow this rubric: Summary, Issues, Strengths.",
        },
        {
            "role": "user",
            "content": f"Review this PR diff:\n\n{diff}",
        },
    ],
    max_tokens=512,
)

print(response.choices[0].message.content)