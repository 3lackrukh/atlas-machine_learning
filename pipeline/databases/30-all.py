#!/usr/bin/env python3
""" 30-all.py: Lists all documents in a MongoDB collection """

def list_all(mongo_collection):
    """Lists all documents in a collection"""
    return [doc for doc in mongo_collection.find()] 