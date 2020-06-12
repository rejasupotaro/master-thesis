FROM python:3.8.3-buster

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace
COPY pyproject.toml .
COPY poetry.lock .
COPY src src

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH="${PATH}:/root/.poetry/bin"
RUN poetry install

RUN mkdir data
RUN mkdir data/external
RUN mkdir data/interim
RUN mkdir data/processed
RUN mkdir data/raw
RUN mkdir models
RUN mkdir logs
RUN mkdir logs/fit

COPY .env .
