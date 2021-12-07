savedir=$(pwd)

# cd to runs-dir
cd $(git rev-parse --show-toplevel)/data
mkdir -p ../runs

# Unzip all dirs in runs in parallel
for run in *.zip; do
	unzip -o "${run}" -d "../runs/" &
done; wait

cd $savedir
