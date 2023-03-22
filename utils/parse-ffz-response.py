#%%
import json
import pymongo
import requests

# %%
streamer = "global"
url="https://api.betterttv.net/3/cached/emotes/global"
data = requests.get(url).json()
# %%
filtered = list(
    map(
        lambda x: {'stream': streamer, 'emote': x['code']}
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
