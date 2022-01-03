# Neuralcollapse project
For MIT course 9.520 Statistical Learning Theory and Applicatoins, Fall 2021

Run an experiment by
`./do_experiment.sh path/to/config_file.yaml`
which will (1) run the training procedure, (2) do the relevant measurements and (3) concatenate the config-file and the output-plots into a single file.

To run the NC measurements in the second-to-last layer, run
`python3 src/do_measurements.py -cfg path/to/config_file.yaml -stl ; ./scripts/parse_fifs.sh -cfg path/to/config_file.yaml -stl` after having run either `./do_experiment.sh path/to/config_file.yaml` or `python3 src/do_training.py -cfg path/to/config_file`

