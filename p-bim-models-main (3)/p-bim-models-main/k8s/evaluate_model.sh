#!/bin/bash
set -e
RUN_ID="$1"
NAME="$2"
PROJECT="$3"
cat k8s/evaluate_model_job_template.yml |\
sed "s/{{RUN_ID}}/$RUN_ID/g" |\
sed "s/{{NAME}}/$NAME/g" |\
sed "s/{{PROJECT}}/$PROJECT/g" |\
kubectl apply -f -