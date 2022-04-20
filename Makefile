all: build run

build: 
	sudo docker build -t prominendt/jetson-containers-python-blob_detection:latest .

run:
	sudo docker run --privileged --net host -p 5000:5000 -p 6000:6000 --env NUMBER_OF_DETECTION_THREADS=1 --env DISABLE_JIT=True -v /dev/bus/usb:/dev/bus/usb prominendt/jetson-containers-python-blob_detection:latest