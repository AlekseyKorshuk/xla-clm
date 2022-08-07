#!/bin/bash

set -e

export IMAGE=gcr.io/chai-959f8/training:tf-inference-image
docker build --platform linux/amd64 --cache-from $IMAGE -t $IMAGE .
docker push $IMAGE
kubectl delete inferenceservice tf-inference
kubectl apply -f opt_inference.yaml