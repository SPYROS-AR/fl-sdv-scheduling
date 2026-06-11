FROM ghcr.io/eclipse-sumo/sumo:latest

ENV PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH:-}"

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt


COPY . /app

CMD ["python3", "./src/main.py"]
