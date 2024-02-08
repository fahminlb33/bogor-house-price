# pull official base image
FROM python:3.11.4-slim-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# install dependencies
RUN pip install --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt

# copy project
COPY . .

# start app
CMD ["gunicorn", "--workers", "4", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "app:create_app()"]
