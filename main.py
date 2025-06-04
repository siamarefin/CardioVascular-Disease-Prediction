from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.code import predict

app = FastAPI()

# Allow CORS for your Next.js frontend (default: http://localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Frontend domains allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """
    try:
        result = predict(input_data)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
