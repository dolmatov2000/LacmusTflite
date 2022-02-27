FROM tensorflow/tensorflow:2.7.0-gpu-jupyter 

RUN apt update

RUN pip install -U albumentations

