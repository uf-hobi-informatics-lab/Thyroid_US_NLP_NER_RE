#!/bin/sh

while getopts c:e:n: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        e) exper=${OPTARG};;
        n) nodes=${OPTARG};;
    esac
done

# python run_ner.py --config $config --experiment $exper --gpu_nodes $nodes
# python run_relation.py --config $config --experiment $exper --gpu_nodes $nodes
python run_post_processing.py --config $config --experiment $exper --gpu_nodes $nodes