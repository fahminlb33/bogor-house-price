# pull official base image
FROM python:3.11.9-bookworm

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONUNBUFFERED=1

# install python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose port
EXPOSE 8000

# start app
CMD ["gunicorn", "--workers", "2", "--certfile", "/etc/ssl/cert.pem", "--keyfile", "/etc/ssl/key.pem", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "app:create_app()"]
