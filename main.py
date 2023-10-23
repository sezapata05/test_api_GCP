from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"msg" : "Espero que si funcione!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)