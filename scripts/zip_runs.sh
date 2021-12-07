savedir=$(pwd)

# cd to runs-dir
cd $(git rev-parse --show-toplevel)/runs
mkdir -p ../data

# Zip all dirs in runs in parallel
for run in */; do
	zip -0 -r "../data/${run%/}.zip" "$run" &
done; wait

cd $savedir
