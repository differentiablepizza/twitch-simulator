#%%
import pandas as pd
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    GPT2Tokenizer
    )
from transformers.models.gpt_neo import GPTNeoForCausalLM
from torch.utils.data import DataLoader
from torch import nn
from glob import glob
from datetime import datetime, timedelta
from tqdm import tqdm
from emote_detection import get_emotes
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:40"

#%%
# Set up model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
suffix = model_name.split("-")[-1]
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPTNeoForCausalLM.from_pretrained(model_name)
model = model.to("cuda")

#%%
# Load data (Garbage is already filtered in the MongoDB pipeline)
path="../data/processed_chatlogs/filian/all-in-2023-04-01.csv"
df = pd.concat([pd.read_csv(f, sep=',', index_col=0).reset_index(drop=True) for f in glob(path)]).reset_index(drop=True)
# Remove duplicates
df = df.drop_duplicates(subset=["message", "username"])
df["username"] = df["username"].apply(lambda x: "STREAMER" if (x == "[STREAMER]") else ("STREAM TITLE" if (x=="[STREAM TITLE]") else "CHAT"))
# Drop \n
df["message"] = df["message"].str.replace("\n", " ")

#%%
df.head()

#%%
#Stats
df.info()

#%%
#Augmentation
window_size = 10
data = []
def window(n=10):
    window_size = 10
    d = []
    for i in tqdm(range(len(df)-window_size), desc="Processing data"):
        block = df.iloc[i:i+window_size].copy()
        block["text"] = block["username"] + ": " + block["message"]
        block = block["text"].tolist()
        d.append("\n".join(block))
    return d
data = window(10) + window(5) + window(2) + window(1)
raw_data = pd.DataFrame(data, columns=["text"])
#Jumble
raw_data = raw_data.sample(frac=1).reset_index(drop=True)
raw_data.to_csv("data/raw_data.csv", index=False, header=False)

#%%
# Define training dataset and dataloader
train_dataset = TextDataset(
    file_path="data/raw_data.csv",
    tokenizer=tokenizer,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=True
)

# Define loss and optimizer
training_args = TrainingArguments(
    output_dir="models",
    overwrite_output_dir=True,
    num_train_epochs=5,
    auto_find_batch_size=True,
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=1,
    no_cuda=False,
    fp16=True
)

#%%
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

#%%
trainer.train()

#%%
# Upload
# Requires setting up HuggingFace credentials
tokenizer.push_to_hub(f"twitch-simulator-filian-neo-{suffix}", commit_message="Add toknizer")
model.push_to_hub(f"twitch-simulator-filian-neo-{suffix}", commit_message="Add model")

#%%
# Check if a model is already saved with name "gpt_neo_chatbot_vj", where j is the version number. If so, save the model with the next version number.
trainer.save_model(f"models/gpt_neo_chatbot_v{len(glob('models/gpt_neo_chatbot_v*'))}")
tokenizer.save_pretrained(f"tokenizers/gpt_neo_chatbot_v{len(glob('tokenizers/gpt_neo_chatbot_v*'))}")

#%%
# Generate text
prompt = """
STREAMER: I am a
"""
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# %%
