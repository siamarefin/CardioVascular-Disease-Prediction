from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from api.code import  predict
import os
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
from fastapi.responses import FileResponse


app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name = "static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/files", StaticFiles(directory="files"), name="files")
 
# Serve home.html 

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/home.html", "r", encoding="utf-8") as file:
        return file.read()
    





@app.post("/predict")
def make_prediction(input_data: dict):
    """
    json_input = {
    "ap_hi": 120,
    "ap_lo": 80,
    "cholesterol": 1, 
    "age_years": 47,  
    "bmi": 26.573129         
}

    API endpoint to call the predict function from code.py
    :param input_data: JSON with input features
    :return: Prediction result
    """
    return predict(input_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)


    
