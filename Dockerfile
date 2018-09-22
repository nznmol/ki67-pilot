FROM tensorflow/tensorflow:latest-gpu-py3
COPY vendor/tf_unet /tf_unet
RUN pip install -e /tf_unet
