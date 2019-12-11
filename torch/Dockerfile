FROM pytorch/pytorch
RUN pip install --upgrade pip
COPY requirements.txt /etc/requirements.txt
RUN pip install -r /etc/requirements.txt
WORKDIR /share
CMD jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=''