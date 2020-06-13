FROM python:3.8.3-buster

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace
COPY pyproject.toml .
COPY poetry.lock .

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
ENV PATH="${PATH}:/root/.poetry/bin"

COPY src src

RUN poetry install

RUN mkdir models
RUN mkdir logs
RUN mkdir logs/fit

COPY .env .
