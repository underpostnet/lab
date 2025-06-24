sudo podman pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
sudo rm -rf ../lab/spark-cuda_latest.tar
node bin dockerfile-image-build --path ../lab/src/spark/check_gpu --image-name=spark-cuda:latest --image-path=../lab --podman-save --kind-load --no-cache
kubectl delete SparkApplication spark-gpu-test
kubectl delete Role spark-pod-creator
kubectl delete RoleBinding spark-default-role-binding
kubectl apply -f ./manifests/deployment/spark/spark-rbac.yaml
sleep 5
kubectl apply -f ./manifests/deployment/spark/cuda-test.yaml
kubectl get SparkApplication -o wide -w
