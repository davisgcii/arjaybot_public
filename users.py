from dotenv import load_dotenv
import pymongo
import os

load_dotenv()

client = pymongo.MongoClient(os.environ["MONGODB_URI"])
db = client[os.environ["MONGODB_NAME"]]

users = db["users"]


def add_user(user_id, doc_ids=[], queries=[], chats=[], agreed_to_terms=False):
    result = users.insert_one(
        {
            "user_id": user_id,
            "doc_ids": doc_ids,
            "queries": queries,
            "chats": chats,
            "agreed_to_terms": agreed_to_terms,
        }
    )
    return result


def update_agreement(user_id):
    result = users.update_one({"user_id": user_id}, {"$set": {"agreed_to_terms": True}})
    return result


def add_query(user_id, query):
    result = users.update_one({"user_id": user_id}, {"$push": {"queries": query}})
    return result


def add_chat(user_id, chat):
    result = users.update_one({"user_id": user_id}, {"$push": {"chats": chat}})
    return result


def user_agrees_to_terms(user_id):
    user = users.find_one({"user_id": user_id})

    if user is None:
        add_user(user_id)
        return False

    return user["agreed_to_terms"]


def add_doc_to_user(user_id, doc_id):
    result = users.update_one({"user_id": user_id}, {"$push": {"doc_ids": doc_id}})
    return result
