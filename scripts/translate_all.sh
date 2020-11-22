

for experiment in `ls checkpoints`
do
	dataset=`echo $experiment | cut -f1 -d "-"`	
	config=`echo $experiment | cut -f2 -d "-"`	
	for checkpoint in `ls -v checkpoints/$experiment`
	do
		step=`echo $checkpoint | cut -f2 -d "."`
		save_path="translations/$experiment/$step"
		ckp_path=checkpoints/$experiment/$checkpoint
		mkdir -p translations/$experiment
		if [ ! -f $save_path ];
		then
			bash scripts/translate.sh $dataset $ckp_path $config
		fi
	done
done
