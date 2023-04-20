#!/bin/bash

STRATEGIES=("hourly" "uniform" "weighted-random")
AGGREGATIONS=("mean" "interpolate" "nosampling")
SCENARIO="S3"

for strategy in ${STRATEGIES[@]}; do
    for agg in ${AGGREGATIONS[@]}; do
	name="${agg}-${strategy}"
	python k8s/evaluate_model.py \
		--job-template jobs/evaluate/pbim/gan/$SCENARIO/$name.yml \
		--name $name \
		--project $name \
		--ids ""
    done
done
