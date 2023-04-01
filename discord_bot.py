
#%%
#Discord bot to interact with the model
import discord
from transformers import GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
import re
import os
from glob import glob
import asyncio
import sys
import torch
import logging
import logging.handlers

logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
logging.getLogger('discord.http').setLevel(logging.INFO)

handler = logging.handlers.RotatingFileHandler(
    filename='discord.log',
    encoding='utf-8',
    maxBytes=32 * 1024 * 1024,  # 32 MiB
    backupCount=5,  # Rotate through 5 files
)
dt_fmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter('[{asctime}] [{levelname:<8}] {name}: {message}', dt_fmt, style='{')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
#Get latest model
models = glob("models/gpt_neo_chatbot_v*")
model_name = max(models, key=os.path.getctime)
model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# %%

#%%
def prompt(messages):
    # If the messase is from the bot, prepend "STREAMER: ", otherwise "CHAT: "
    # The messages come in reverse, the earliest ones are the first ones.
    prompt = []
    char_count = 0
    for message in messages:
        if message.author == client.user:
            prompt.append(f"STREAMER: {message.content}")
        else:
            prompt.append(f"CHAT: {message.content}\n")
        #Check is mesage is more than 500 characters
        char_count += len(message.content)
        if len(prompt) > 500:
            break
    prompt.reverse()
    prompt.append("STREAMER: ")
    prompt = "\n".join(prompt)
    logger.debug(f"Input:\n{prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, attention_mask = attention_mask, max_length=1000, do_sample=True, top_k=5, top_p=0.95, temperature=0.9)
    message = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.debug(f"Output:\n{message}")
    #Extracting last response with "STREAMER:" prefix
    message = message.split("\n")
    message = [line for line in message if line.startswith("STREAMER:")][-1]
    message=re.sub("STREAMER: ","",message)
    message=message.strip().strip("\"").strip("\n")
    return message
    

#%%
# Create a new Intents object
intents = discord.Intents.default()

# Enable the events that your bot will handle
intents.members = True
intents.messages = True

# Create a new client with the Intents object
client = discord.Client(intents=intents)

#%%
@client.event
async def on_ready():
    print("Ready")
@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return
    
    response = ""
    if message.content.startswith("!reset"):
        response = prompt([])
        logger.debug(f"Resetting history")
    else:
        history = []
        async for message in message.channel.history(limit=10):
            history.append(message)
        #Drop commands
        history = [message for message in history if not message.content.startswith("!")]
        logger.debug(f"History:\n{history}")
        response = prompt(history)

    # Check if the message mentions the bot
    if client.user in message.mentions:
        # Send a reply
        await message.channel.send(response)

    # Check if the message was sent in a direct message channel
    if isinstance(message.channel, discord.DMChannel):
        # Send a reply
        await message.channel.send(response)

#%%
if __name__=="__main__":
    TOKEN=os.environ["DISCORD_BOT_TOKEN"]
    client.run(TOKEN, log_handler=None)