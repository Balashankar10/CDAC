from pymongo import MongoClient
import gridfs
import os
from bson import ObjectId
import hashlib
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Connect to MongoDB Atlas
mongo_url = os.getenv("MONGO_URL")
client = MongoClient(mongo_url)
db = client["phishingDB"]
col = db["phishing_calls"]
fs = gridfs.GridFS(db)
hash_col = db["audio_hashes"]


def get_existing_rating(number):
    record = col.find_one({"number": number})
    return record.get("rating") if record else None

def hash_audio_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_audio_file(file_path, filename):
    file_hash = hash_audio_file(file_path)

    # Check if hash exists
    existing = hash_col.find_one({"hash": file_hash})
    if existing:
        raise ValueError("Duplicate audio file detected. Illegal entry.")

    with open(file_path, "rb") as f:
        audio_id = fs.put(f, filename=filename)

    hash_col.insert_one({
        "hash": file_hash,
        "audio_id": audio_id,
        "upload_count": 1
    })
    return str(audio_id), 1

def save_rating(number, transcript, new_rating, audio_id, phishing_percent):
    update_doc = {
        "$set": {
            "rating": new_rating
        },
        "$push": {
            "transcript": {
                "$each": [transcript]
            },
            "audio_records": {
                "phishing_percent": phishing_percent,
                "audio_id": audio_id
            }
        }
    }
    col.update_one({"number": number}, update_doc, upsert=True)

def get_full_record(number):
    return col.find_one({"number": number})

def get_audio_file(audio_id):
    return fs.get(ObjectId(audio_id)).read()
