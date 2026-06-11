FROM ghcr.io/eclipse-sumo/sumo:latest

ENV PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH:-}"

RUN pip3 install traci sumolib

WORKDIR /app

COPY . /app

CMD ["python3", "./src/main.py"]
