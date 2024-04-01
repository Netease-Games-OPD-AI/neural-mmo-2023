ARG DOCKER_BASE_IMAGE
FROM ${DOCKER_BASE_IMAGE}

COPY requirements.txt ./

RUN sudo rm /etc/apt/sources.list.d/* && sudo apt-get update && sudo apt-get install -y vim

RUN $HOME/anaconda3/bin/pip --no-cache-dir install -r requirements.txt

RUN sudo rm requirements.txt
