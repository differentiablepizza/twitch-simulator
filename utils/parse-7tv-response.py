#%%
import json
import pymongo
import requests

# %%
url="https://api.7tv.app/v2/users/filian/emotes"
streamer = "filian"
data = requests.get(url).json()
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
