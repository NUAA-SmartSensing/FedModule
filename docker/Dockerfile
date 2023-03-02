FROM python:3.8
RUN git clone https://github.com/desperadoccy/async-FL.git
WORKDIR /async-FL
RUN pip install -r requirements.txt
WORKDIR /async-FL/src/fl
ENTRYPOINT ["python", "main.py"]