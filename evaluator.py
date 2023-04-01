#%%
from transformers import GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
import re

#%%
# Set up model and tokenizer
model_name = "models/gpt_neo_chatbot_v6"
tokenizer_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# %%
# Generate text
prompt = """
CHAT: Hello, how are you?
STREAMER: 
"""
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=20, top_p=0.95, temperature=0.7)
message = tokenizer.decode(output[0], skip_special_tokens=True)
#Extracting only STREAMER response
r=r"STREAMER:(.*)"
message=re.findall(r,message,flags=re.DOTALL)[0].split("CHAT")[0]
message=message.strip().strip("\"").strip("\n")
print(message)
# %%
