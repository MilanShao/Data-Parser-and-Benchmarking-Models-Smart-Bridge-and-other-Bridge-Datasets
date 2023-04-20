#!/bin/bash
PREFIX="$1"
NAME="$2"
PREFIX_ESCAPED="${PREFIX//\//-}"

cat k8s/train_model_job_template.yml |\
sed "s|{{PREFIX}}|$PREFIX|g" |\
sed "s|{{PREFIX_ESCAPED}}|$PREFIX_ESCAPED|g" |\
sed "s/{{NAME}}/$NAME/g" |\
sed "s/{{RUN_NAME_SUFFIX}}//g" |\
kubectl apply -f -