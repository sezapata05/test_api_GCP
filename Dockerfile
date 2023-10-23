# Usa una imagen base de Python
FROM python:3.11

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de tu proyecto al directorio de trabajo
RUN echo "Copy"
COPY . .

# Instala las dependencias de tu proyecto
RUN echo "Installing dependencies"
# RUN pip install -U -r requirements_final.txt
RUN pip install --no-cache-dir --upgrade -r requirements_final.txt


# Expone el puerto en el que se ejecutará tu aplicación FastAPI
EXPOSE 8000

# Ejecuta la aplicación FastAPI con Uvicorn
# CMD ["uvicorn", "__init__:application", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["uvicorn", "challenge.__init__:application", "--host", "0.0.0.0", "--port", "8000", "--reload"]


