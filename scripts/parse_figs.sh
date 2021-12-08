#! /bin/env bash

start_dir=$(pwd)

savedir=$( grep "  save-dir:" $1 | awk '{print $2}' )
echo Using files in $(git rev-parse --show-toplevel)/$savedir
# cd to latex-dir
cd $(git rev-parse --show-toplevel)/scripts/figure_parser

# Create symlink to run
rm runlink
ln -sf ../../$savedir runlink

echo
echo
pdflatex -shell-escape main.tex
echo
echo

cp main.pdf runlink/output.pdf

echo Finished, used files in $(git rev-parse --show-toplevel)/$savedir

cd $start_dir
