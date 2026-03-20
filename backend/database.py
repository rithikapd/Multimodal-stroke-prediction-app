import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred,{
        "databaseURL":"https://multimodal-stroke-prediction-default-rtdb.firebaseio.com/"
    })


def save_result(data):

    ref = db.reference("predictions")
    ref.push(data)


def get_all_results():

    ref = db.reference("predictions")
    return ref.get()