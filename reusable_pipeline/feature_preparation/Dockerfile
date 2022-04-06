FROM python:3.9.4
WORKDIR /feature_prep
COPY . /feature_prep
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "feature_preparation.py"]