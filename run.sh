#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
prefix_markers=$2  # e.g. data/markers/tls/

python get_marker_lists.py $prefix_markers
bash get_marker_scores.sh $prefix
python phenotype.py ${prefix}markers/phenotype/
