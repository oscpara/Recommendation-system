FROM python:3.10-slim


WORKDIR src/code/app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY src/code app/src/code 

# Run your app (adjust if your entrypoint is different)
CMD ["python", "app.py"]