from ctransformers import AutoModelForCausalLM
import chainlit as cl

# Commande à runner initialement
# chainlit run chat.py -w

# !!!! Le modèle va être mis en cache dans ~/.cache/huggingface/hub/ (si besoin d'espace disque)

history = []

def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system="You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}, Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt

"""@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.svg",
            )
    ]"""

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
    print("A new chat session has started!")


"""history = []
answer = ""





question = "Which city the capital of India?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "And which is of the United States?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)"""
