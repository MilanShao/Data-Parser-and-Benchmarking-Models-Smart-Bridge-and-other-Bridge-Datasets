#!/bin/bash
PREFIX="$1"
NAME="$2"
COUNT="$3"
PREFIX_ESCAPED="${PREFIX//\//-}"
for i in $(seq 1 $COUNT); do
  RUN_NAME_SUFFIX="-$i"
  echo "Processing $NAME-$i"
  cat k8s/train_model_job_template.yml | \
  sed "s|{{PREFIX}}|$PREFIX|g" | \
  sed "s|{{PREFIX_ESCAPED}}|$PREFIX_ESCAPED|g" | \
  sed "s/{{NAME}}/$NAME/g" | \
  sed "s/{{RUN_NAME_SUFFIX}}/$RUN_NAME_SUFFIX/g" | \
  kubectl apply -f -
done