FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY app/main.py /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]


