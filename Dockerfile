FROM python:3.10-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run your app (adjust if your entrypoint is different)
CMD ["python", "src/code/app.py"]