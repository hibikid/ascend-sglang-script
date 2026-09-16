curl -v http://127.0.0.1:6699/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.1-w4a8",
    "messages": [
      {
        "role": "user",
        "content": "A box contains 25 pencils. If 7 pencils are given away and the remaining pencils are divided equally among 3 students, how many pencils does each student get? Please explain your solution step by step."
      }
    ],
    "temperature": 0,
    "max_tokens": 128
  }'