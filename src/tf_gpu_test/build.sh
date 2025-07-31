PROJECT_DIR=/home/dd/lab/src/tf_gpu_test
PROJECT_VERSION=0.0.11
IMAGE_NAME=tf_gpu_test

IMAGE_NAME_FULL="${IMAGE_NAME}:${PROJECT_VERSION}"

cd ${PROJECT_DIR}

sudo podman pull docker.io/tensorflow/tensorflow:2.13.0-gpu

sudo rm -rf ${PROJECT_DIR}/${IMAGE_NAME}_${PROJECT_VERSION}.tar

underpost dockerfile-image-build --path ${PROJECT_DIR} --image-name=${IMAGE_NAME_FULL} --image-path=${PROJECT_DIR} --podman-save --kubeadm-load --reset

kubectl apply -f ./${IMAGE_NAME}.yaml

kubectl get sparkapplication -w -o wide