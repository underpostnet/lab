sudo podman pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
sudo rm -rf ../lab/spark-cuda_latest.tar
node bin dockerfile-image-build --path ../lab/src/spark/check_gpu --image-name=spark-cuda:latest --image-path=../lab --podman-save --kind-load --no-cache
kubectl apply -f ./manifests/deployment/spark/cuda-test.yaml
kubectl get pods -w -o wide
