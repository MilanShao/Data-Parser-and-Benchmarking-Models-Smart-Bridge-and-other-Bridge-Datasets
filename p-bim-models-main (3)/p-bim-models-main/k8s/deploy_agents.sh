#!/bin/bash
NAME="$1"
SWEEP_ID="$2"
COUNT="${3:-1}"
for i in $(seq 1 $COUNT); do
  echo "Deploying agent $i..."
  cat k8s/sweep_agent_job_template.yml |\
  sed "s|{{NAME}}|$NAME|g" |\
  sed "s|{{SWEEP_ID}}|$SWEEP_ID|g" |\
  sed "s|{{AGENT_ID}}|$i|g" |\
  kubectl apply -f -
done
