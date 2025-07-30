PROJECT_DIR=/home/dd/lab/src/tf_gpu_test
PROJECT_VERSION=0.0.11
IMAGE_NAME=tf_gpu_test

IMAGE_NAME_FULL="${IMAGE_NAME}:${PROJECT_VERSION}"

cd ${PROJECT_DIR}

sudo podman pull nvidia/cuda:12.9.0-base-ubuntu24.04

sudo rm -rf ${PROJECT_DIR}/${IMAGE_NAME}_${PROJECT_VERSION}.tar

underpost dockerfile-image-build --path ${PROJECT_DIR} --image-name=${IMAGE_NAME_FULL} --image-path=${PROJECT_DIR} --podman-save --kubeadm-load --no-cache

kubectl apply -f ./${IMAGE_NAME}.yaml

kubectl get sparkapplication -w -o wide