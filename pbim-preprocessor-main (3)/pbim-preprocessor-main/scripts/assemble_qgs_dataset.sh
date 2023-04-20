#!/bin/bash
INPUT_PATH=$1
OUTPUT_PATH=$2

find $INPUT_PATH -type f -name "*.TXT" -print0 | while IFS= read -r -d $'\0' file;
do
  filename=${file##*/}
  filename=${filename%.*}
  anomalous=${filename:3:2}
  if [ $anomalous == "AU" || $anomalous == "BU" ]; then
    anomalous="False"
  else
    anomalous="True"
  fi
  echo "Processing scenario $filename (anomalous: $anomalous)"
  python bin/main.py assemble grandstand $INPUT_PATH "$OUTPUT_PATH/$filename.dat" --scenario $filename --output-format binary --is-anomalous $anomalous
done