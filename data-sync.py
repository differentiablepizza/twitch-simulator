import pymongo
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import argparse
import logging
import traceback

#Set to STDOUT
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Sync pipeline with respect to the chat collection
aggregate_pipeline = [
    {
        "$unwind": "$subtitles"
    },
    {
        "$project": {
            "_id": 0,
            "username": {"$literal": "[STREAMER]"},
            "text": "$subtitles.text",
            "start": "$subtitles.start",
            "end": "$subtitles.end"
        }
    },
    {
        "$unionWith": {
            "coll": "filian",
            "pipeline": [
                {
                    "$project": {
                        "_id": 0,
                        "username": "$username",
                        "text": "$text",
                        "start": "$timestamp"
                    }
                }
            ]
        }
    },
    {
        "$project": {
            "username": 1,
            "text": 1,
            "start": 1
        }
    }
]
class SyncProcedure:
    """
    Syncs chat messages with VODs captions from a given start time.
    """
    def __init__(self, caption_collection, chat_collection, streamer_tag = "[STREAMER]", chat_tag = "[CHAT]") -> None:
        self.caption_collection = caption_collection
        self.chat_collection = chat_collection
        self.streamer_tag = streamer_tag
        self.chat_tag = chat_tag
    def query(self, start_time: datetime = None, end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        Queries chat messages and VOD captions in chronological order by using a single aggregate pipeline on the chat collection.
        """
        self.aggregate_pipeline = aggregate_pipeline.copy()
        match_pipeline = {"$match": {"start": None}}
        if start_time is not None:
            match_pipeline["$match"]["start"] = {"$gte": start_time}
        if end_time is not None:
            match_pipeline["$match"]["start"] = {"$lte": end_time}
        if match_pipeline["$match"]["start"] is not None:
            self.aggregate_pipeline.append(match_pipeline)
        self.aggregate_pipeline.append({"$sort": {"start": 1}})
        self.aggregate_cursor = self.chat_collection.aggregate(self.aggregate_pipeline)
        #Yield streamer and chat messages in chronological order
        logger.debug("Iterating through chat messages and captions")
        self.aggregate_cursor
    def save(self, target_collection, name) -> None:
        """
        Saves the synchronization result to the target database.
        """
        self.target_collection = target_collection
        logger.debug("Saving synchronization result to database")
        self.target_collection.insert_many(self.aggregate_cursor)

def get_start_sync_from_chat_collection(chat_collection):
    pipeline_earliest = [
        {
            "$project": {
                "timestamp": 1,
            }
        },
        {
            "$sort": {
                "timestamp": 1
            }
        },
        {
            "$limit": 1
        }
    ]
    start_sync = chat_collection.aggregate(pipeline_earliest).next()
    start_sync = start_sync['timestamp']
    return start_sync

def get_end_sync_from_chat_collection(chat_collection):
    pipeline_latest = [
        {
            "$project": {
                "timestamp": 1,
            }
        },
        {
            "$sort": {
                "timestamp": -1
            }
        },
        {
            "$limit": 1
        }
    ]
    end_sync = chat_collection.aggregate(pipeline_latest).next()
    end_sync = end_sync['timestamp']
    return end_sync

def get_last_sync_from_target(target_collection):
    pipeline_latest = [
        {
            "$project": {
                "start": 1,
            }
        },
        {
            "$sort": {
                "start": -1
            }
        },
        {
            "$limit": 1
        }
    ]
    last_sync = target_collection.aggregate(pipeline_latest).next()
    last_sync = last_sync['end_time']
    return last_sync

if __name__=="__main__":
    uri = 'mongodb://localhost:27017'
    args = argparse.ArgumentParser()
    args.add_argument("--chat_database", type=str, default="twitch-stream", help="The name of the database containing the chat messages")
    args.add_argument("--caption_database", type=str, default="twitch-subtitles", help="The name of the database containing the VOD captions")
    args.add_argument("--target_database", type=str, default="twitch-synced-data", help="The name of the database to save the synchronization result")
    args.add_argument("--overwrite", action="store_true", help="Overwrite the existing synchronization result")
    parsed = args.parse_args()

    client = pymongo.MongoClient(uri)
    chat_database = client[parsed.chat_database]
    caption_database = client[parsed.caption_database]
    target_database = client[parsed.target_database]

    if parsed.overwrite:
        logger.warning("Deleting database %s", parsed.target_database)
        client.drop_database(parsed.target_database)

    # Match collection names from the chat database and the caption database, if they don't match, skip
    for caption_collection_name in caption_database.list_collection_names():
        logger.debug("Processing collection %s", caption_collection_name)
        if "view" in caption_collection_name:
            logger.debug("Skipping view collection")
            continue
        try:
            chat_collection = chat_database[caption_collection_name]
            caption_collection = caption_database[caption_collection_name]
            start_sync = None
            # Check the most recent sync document in the target database
            if not parsed.overwrite:
                if caption_collection_name in target_database.list_collection_names():
                    start_sync = get_last_sync_from_target(target_database[caption_collection_name])
                    logger.debug(f"Syncing from the last synchronization point ({start_sync=})")
                else:
                    # Get start and end times from the chat database
                    start_sync = get_start_sync_from_chat_collection(chat_collection)
                    logger.debug(f"Syncing from the beginning of the chat log ({start_sync=})")
            else:
                start_sync = get_start_sync_from_chat_collection(chat_collection)
                logger.debug(f"Overwrite flag is set, syncing from the beginning of the chat log ({start_sync=})")
            end_sync = get_end_sync_from_chat_collection(chat_collection)
            logger.debug(f"Syncing until the end of the chat log ({end_sync=})")
            sync = SyncProcedure(caption_collection, chat_collection)
            logger.info(f"Syncing {caption_collection_name} from {start_sync} to {end_sync}")
            sync.query(
                start_time=start_sync,
                end_time=end_sync
            )
            init_tstamp = start_sync.strftime("%Y-%m-%d_%H:%M:%S")
            target_tstamp = end_sync.strftime("%Y-%m-%d_%H:%M:%S")
            sync.save(
                target_collection=target_database[caption_collection_name],
                name=f"{init_tstamp}_{target_tstamp}"
            )
        except Exception as e:
            # Print complete error stack with traceback
            traceback.print_exc()
            continue