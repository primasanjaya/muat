#build docker for linux amd architecture 
docker build --platform linux/amd64 -t muat:latest .

#run and test it
docker run -it muat:latest muat -h