all: build run


build: 
	docker build -t blob-app .

run:
	docker run --privileged -p 5000:5000 -v /dev/bus/usb:/dev/bus/usb blob-app