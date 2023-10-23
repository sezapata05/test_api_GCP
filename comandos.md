## Docker

### Build
docker build -t mi-aplicacion-fastapi:latest .

### Run
docker run -d -p 8000:8000 mi-aplicacion-fastapi:latest

### Run in local
uvicorn challenge.__init__:application --reload