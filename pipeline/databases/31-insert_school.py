#!/usr/bin/env python3
""" 31-insert_school.py: Inserts a new document in a MongoDB collection """

def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection based on kwargs"""
    return mongo_collection.insert_one(kwargs).inserted_id 