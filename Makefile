all: build run

build: 
	docker build -t blob-app .

run:
	docker run --privileged -p 5000:5000 --env NUMBER_OF_DETECTION_THREADS=1 --env DISABLE_JIT=True -v /dev/bus/usb:/dev/bus/usb blob-app