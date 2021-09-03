FROM python:latest

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv

COPY motion_detect.py /script/motion_detect.py
RUN chmod +x /script/motion_detect.py

ENTRYPOINT ["/script/motion_detect.py"]
