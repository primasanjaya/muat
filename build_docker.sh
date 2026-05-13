#build docker for linux amd architecture 
docker build --platform linux/amd64 -t muat:v0.1.20 .

#run and test it
#docker run -it muat:v0.1.20 muat -h
