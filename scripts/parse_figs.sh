#! /bin/env bash

start_dir=$(pwd)

savedir=$( grep "save-dir:" $1 | awk '{print $2}' )
echo Using files in $(git rev-parse --show-toplevel)/$savedir
# cd to latex-dir
cd $(git rev-parse --show-toplevel)/scripts/figure_parser

# Create symlink to run
rm runlink
ln -sf ../../$savedir runlink

pwd
echo $savedir
# Compile pdf with measurement-plots:
cp filepaths.csv _latex_plot_paths.csv
echo
echo
pdflatex -shell-escape main.tex
echo
echo
cp main.pdf runlink/output.pdf
echo

# If there are measurements from second-to-last layer then plot them
if [[ -d runlink/stl_measurements ]]
then
	cp stl_filepaths.csv _latex_plot_paths.csv
	echo "Parsing files in stl_measurements:"
	echo
	pdflatex -shell-escape main.tex
	echo
	echo
	cp main.pdf runlink/stl_output.pdf
	echo Finished parsing stl_measuremetns
	echo
fi

echo Finished, used files in $(git rev-parse --show-toplevel)/$savedir

cd $start_dir
