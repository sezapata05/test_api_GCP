from challenge.api import app


application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

# uvicorn __init__:application --reload