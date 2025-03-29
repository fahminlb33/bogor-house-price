# pull official base image
FROM python:3.11.8-slim

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONUNBUFFERED=1

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
	curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# update pip
RUN pip install --upgrade pip

# install python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose port
EXPOSE 8000

# add health check
HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

# start app
CMD ["streamlit", "run", "Beranda.py", "--server.port=8000", "--server.address=0.0.0.0"]
