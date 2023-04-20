#!/bin/bash
set -e
RUN_ID="$1"
NAME="$2"
cat k8s/predict_model_job_template.yml |\
sed "s/{{RUN_ID}}/$RUN_ID/g" |\
sed "s/{{NAME}}/$NAME/g" |\
kubectl apply -f -