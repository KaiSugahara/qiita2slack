FROM python:latest
USER root

# UPDATE APT
RUN apt update
RUN apt-get update
RUN apt -y install tzdata
ENV TZ=Asia/Tokyo

# INSTALL LIBRARY(S)
RUN pip3 install --upgrade pip
RUN pip3 install numpy fasttext Flask gunicorn

# CHANGE WORKDIR
WORKDIR /app

# CHANGE CMD
CMD ["gunicorn", "index:app", "-b", "0.0.0.0:80"]