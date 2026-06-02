FROM ghcr.io/eclipse-sumo/sumo:latest

RUN pip3 install traci sumolib

WORKDIR /app

COPY . /app
 
CMD ["python3", "./src/main.py"]
