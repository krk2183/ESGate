# from openai import OpenAI

# client = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key="KEY",
# )

# completion = client.chat.completions.create(
# #   extra_headers={
# #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
# #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
# #   },
#   extra_body={},
#   model="meituan/longcat-flash-chat:free",
#   messages=[
#     {
#       "role": "user",
#       "content": "SYSTEM:<YOU ARE A MODEL THAT GIVES REASONING TO THE GIVEN METRIC PROVIDED BY OUR SERVICE> MESSAGE: Explan why the provided company has a default rate of 0.54"
#     }
#   ]
# )
# print(completion.choices[0].message.content)

