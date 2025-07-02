#!/usr/bin/env python3
""" 34-log_stats.py: Provides stats about Nginx logs stored in MongoDB """

from pymongo import MongoClient

def log_stats():
    """Displays stats about Nginx logs stored in MongoDB"""
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    total = collection.count_documents({})
    print(f"{total} logs")
    print("Methods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")
    status_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_count} status check")

if __name__ == "__main__":
    log_stats() 