#!/bin/bash

start_dir=$(pwd)
config_file=$start_dir/$1
cd $(git rev-parse --show-toplevel)

echo Running experiment with config-file $config_file
date

echo
echo Training nnet:
python3 src/do_training.py --config $config_file
echo
date
echo Generating measurements:
python3 src/do_measurements.py --config $config_file
echo
echo Generating output-file:
./scripts/parse_figs.sh $config_file
echo

savedir=$( grep "  save-dir:" $1 | awk '{print $2}' )
date
echo Finished, output file at $savedir/output.pdf

cd $start_dir
