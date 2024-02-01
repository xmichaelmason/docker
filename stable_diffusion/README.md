```sh
cd stable_diffusion
docker build -t cuda_image -f docker/Dockerfile.cuda .
docker compose up --build -d
```

http://ipaddr:5002/docs