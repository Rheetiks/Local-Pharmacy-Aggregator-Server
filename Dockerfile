FROM python:3.10-slim-buster

WORKDIR /python-docker
RUN apt-get update && apt-get install -y git && apt-get clean
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]