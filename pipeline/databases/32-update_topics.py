#!/usr/bin/env python3
""" 32-update_topics.py: Updates all topics of a school document by name """

def update_topics(mongo_collection, name, topics):
    """Changes all topics of a school document based on the name"""
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}}) 