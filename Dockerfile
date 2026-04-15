FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/server

EXPOSE 8080

CMD ["python", "-c", "import uvicorn; import os; uvicorn.run('main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))"]
