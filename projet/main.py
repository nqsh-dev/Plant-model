from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

model = tf.keras.models.load_model("saine_vs_stress.keras")

IMG_SIZE = (96, 96)

@app.get("/")
def home():
    return {"message": "API saine_vs_stress en marche"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):


    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")


    img = img.resize(IMG_SIZE)


    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)


    img_array = preprocess_input(img_array)


    prediction = model.predict(img_array)

    prob = float(prediction[0][0])


    if prob >= 0.5:
        classe = "stress"
    else:
        classe = "saine"

    return {
        "classe": classe,
        "probabilite": prob
    }