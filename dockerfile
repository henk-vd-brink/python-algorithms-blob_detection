FROM prominendt/icarus-arm64v8-opencv-4.4.0:cuda102a53

RUN apt-get update

WORKDIR /app

ADD requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY app/ app/

CMD ["python3", "-m", "app.app"]
