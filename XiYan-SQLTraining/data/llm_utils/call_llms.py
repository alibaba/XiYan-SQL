from openai import OpenAI


def call_llm():
    client = OpenAI(
        # api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content

