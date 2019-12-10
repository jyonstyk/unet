FROM jjanzic/docker-python3-opencv
RUN apt-get update
RUN apt-get install -y pkg-config gdb graphviz
RUN pip install --upgrade pip
COPY requirements.txt /etc/requirements.txt
RUN pip install -r /etc/requirements.txt
RUN echo 'alias jn="jupyter notebook --ip=0.0.0.0 --allow-root"' >> ~/.bashrc