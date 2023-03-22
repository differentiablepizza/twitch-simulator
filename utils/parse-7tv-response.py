#%%
import json
import pymongo

# %%
filename="/home/user/Downloads/7tv_filian_emotes.json"
streamer = "filian"
with open(filename) as file:
    data = json.load(file)
# %%
filtered = list(
    map(
        lambda x: {'stream': streamer, 'emote': x['name']}
        ,data
        )
    )
# %%
collection_name = "emotes"
database_name = "twitch-stream"
client = pymongo.MongoClient()
# %%
# Upload many
db = client[database_name]
collection = db[collection_name]
collection.insert_many(filtered)
# %%
