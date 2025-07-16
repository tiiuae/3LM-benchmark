#!/bin/bash
ACCOUNT_ID=$(gcloud config get-value project)
BASE_IMG_NAME="eval_image:latest"

if [ $? -ne 0 ]
then
    exit 255
fi


CONTAINER_URI="us-docker.pkg.dev/falcon-training-gpu/tii-eval-arabic/eval_image:latest"
# CONTAINER_URI="us-docker.pkg.dev/falcon-training-gpu/tii-eval-arabic/eval_image:latest"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

echo $CONTAINER_URI

gcloud auth configure-docker us-docker.pkg.dev
if docker build -t $BASE_IMG_NAME .
then
    docker tag $BASE_IMG_NAME $CONTAINER_URI
    docker push $CONTAINER_URI
fi