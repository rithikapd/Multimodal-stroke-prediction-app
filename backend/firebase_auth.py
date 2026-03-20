import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred,{
    "storageBucket":"multimodal-stroke-prediction.appspot.com"
})

bucket = storage.bucket()


def upload_to_firebase(file_path):

    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)

    blob.make_public()

    return blob.public_url