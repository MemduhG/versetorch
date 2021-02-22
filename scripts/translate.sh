
dataset=$1
checkpoint=$2
config=$3

if [ -z "$config" ];
then
	params="--dataset $dataset --checkpoint $checkpoint"
else
	params="--dataset $dataset --checkpoint $checkpoint --config $config"
fi
step=`echo $checkpoint | rev | cut -f1 -d "/" | rev | cut -f1 -d "."`

script_path=~/.scratch/vt-$$.sh
echo "#!/bin/bash -v" > $script_path
echo "#PBS -N ${step}-${dataset}" >> $script_path
cat scripts/val_template.sh >> $script_path
echo $params >> $script_path

echo "Generated script $script_path for checkpoint $checkpoint"
qsub $script_path
