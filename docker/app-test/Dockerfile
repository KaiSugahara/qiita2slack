FROM ubuntu:latest
USER root

# ADD FILES
ADD ./docker/app-test/start.sh /
RUN chmod +rx /start.sh
CMD ["/start.sh"]

# BASH
SHELL ["/bin/bash", "-c"]

# UPDATE & INSTALL
RUN /bin/bash
RUN apt update
RUN apt-get update
RUN apt -y install tzdata
ENV TZ=Asia/Tokyo
RUN apt -y install sudo python3-pip

# CHANGE WORK DIR
WORKDIR /app

# ADD BASHRC
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> $HOME/.bashrc

# INSTALL JUPYTER
RUN pip3 install jupyterlab

# JUPYTER SETTINGS
RUN jupyter lab --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> $HOME/.jupyter/jupyter_lab_config.py
RUN echo "c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_lab_config.py
RUN echo "c.NotebookApp.port = 8888" >> $HOME/.jupyter/jupyter_lab_config.py
RUN echo "c.NotebookApp.token = ''" >> $HOME/.jupyter/jupyter_lab_config.py

EXPOSE 8888