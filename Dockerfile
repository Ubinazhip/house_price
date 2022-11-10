FROM python:3.8.10-slim

RUN pip --no-cache-dir install pipenv

WORKDIR /house_price

COPY ["Pipfile", "Pipfile.lock", "./"]

ENV PYTHONUNBUFFERED=TRUE

RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["*.py", "./"]

WORKDIR /house_price/utils
COPY ["utils", "./"]
WORKDIR /house_price/

RUN apt-get update && apt-get install libgomp1

EXPOSE 9696

#ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"]

ENTRYPOINT ["bash"]
#ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
