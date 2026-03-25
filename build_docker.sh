#build docker for linux amd architecture 
docker build --platform linux/amd64 -t muat:v0.1.17-c .

#run and test it
docker run -it muat:v0.1.17-c muat -h
