from fastapi import FastAPI, File, UploadFile
import uvicorn, requests
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/1")
#"http://localhost:8082/v1/models/potato_disease/versions/2:predict"
endpoint = "http://localhost:8082/v1/models/potato_disease:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/intro")
async def Intro():
    return "Hello World!"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data={

    "instances" : img_batch.tolist()

    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json["predictions"][0])

    #predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        'class  ': predicted_class,
        'confidence  ': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8061)
