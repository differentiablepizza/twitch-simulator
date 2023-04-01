
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

# Load the latest model and tokenizer during initialization
models = glob("models/gpt_neo_chatbot_v*")
model_name = max(models, key=os.path.getctime)
model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def prompt(messages):
    """
    Generate a response to the given list of messages using the loaded GPT-Neo model.

    Args:
        messages: A list of Message objects from a Discord channel.

    Returns:
        A string containing the generated response.
    """
    # Construct the prompt as a list of strings
    query = messages[0].content
    messages = reversed(messages)
    prompt = [f"STREAMER: {message.content}" if message.author == client.user else f"CHAT: {message.content}\n" for message in messages]
    # Limit prompt to maximum of 500 characters
    prompt = prompt[:500]
    # Append the streamer prompt to the end of the list
    prompt.append("STREAMER: ")
    # Join the prompt list using a newline separator
    prompt = "\n".join(prompt)
    logger.debug(f"Prompt:\n{prompt}")
    
    # Encode the prompt as input_ids and attention_mask tensors
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    input_ids = input_ids.to(device)
    
    # Generate a response using the loaded GPT-Neo model
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, do_sample=True, top_k=5, top_p=0.95, temperature=0.9)
    message = tokenizer.decode(output[0], skip_special_tokens=True).replace("\"","")
    organized_message = []
    logger.debug(f"Preview:\n{message}")
    ix = 0
    query_ix = None
    for line in message.split("\n"):
        if line.startswith("STREAMER:"):
            organized_message.append(("STREAMER", line.replace("STREAMER:", "")))
            ix += 1
        elif line.startswith("CHAT:"):
            if query in line:
                organized_message.append(("QUERY", line.replace("CHAT:", "")))
                query_ix = ix
            else:
                organized_message.append(("CHAT", line.replace("CHAT:", "")))
                ix += 1
        else:
            organized_message[-1] = (organized_message[-1][0], organized_message[-1][1] + " " + line)
    
    # Filter only the streamer's response after the query
    logger.debug(f"{query=}\t{query_ix=}")
    message = [line[1].strip().strip("\n").strip("\"") for line in organized_message[query_ix:] if line[0] == "STREAMER"]
    logger.debug(f"Response:\n{message}")

    # Return the response
    message = "\n".join(message)
    
    return message

    

#%%
# Create a new Intents object
intents = discord.Intents.default()

# Enable the events that your bot will handle
intents.members = True
intents.messages = True

# Create a new client with the Intents object
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    """
    Function that gets called when the bot connects to Discord.
    """
    print("Ready")


@client.event
async def on_message(message):
    """
    Function that gets called every time a message is sent in a server or direct message.
    """
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Define a variable to store the bot's response
    response = ""

    if message.content.startswith("!reset"):
        await message.channel.send("IGNORE")

    else:
        # If the message doesn't start with !reset, get the 10 most recent messages in the channel or DM
        history = []
        async for message in message.channel.history(limit=10):
            if message.content != "IGNORE":
                history.append(message)
            else:
                break

        # Remove any messages that start with !
        history = [message for message in history if not message.content.startswith("!")]
        logger.debug(f"History:\n{history}")

        # Get a response based on the history of messages
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