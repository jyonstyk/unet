# 3d u-net
## requirements
```
docker 19.03
latest nvidia-driver
nvidia-docker
```
if you need updating nvidia-docker, read https://github.com/NVIDIA/nvidia-docker

## run docker container
### opencv
```
docker build -t opencv_jupyter cv
docker-compose -f cv/docker-compose.yml up -d -- build
```
after using these commands, access http://localhost:8888 and then connect notebook server

