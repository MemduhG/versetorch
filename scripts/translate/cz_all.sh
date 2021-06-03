# Just adjust these params
model=cz-baseline
language=cz
input=data/cz/cz.dev.src
max_len=256

folder=checkpoints/$model
out_dir=translations/$model
mkdir -p $out_dir

for item in `ls $folder`;
do
	steps=`echo $item | cut -f1 -d "."`
	script=~/.scratch/$steps-$$.sh
	checkpoint=$folder/$item
	output=$out_dir/$steps
    if [ -f $output ];
    then
        continue
    fi
	cat scripts/translate/val_template.sh > $script
	echo "--language $language --max_len $max_len --checkpoint $checkpoint --output $output --input $input" >> $script
	qsub $script
done

out_dir=prose_translations/$model
mkdir -p $out_dir
input=data/$language/prose.txt

for item in `ls $folder`;
do
        steps=`echo $item | cut -f1 -d "."`
        script=~/.scratch/$language-pr-$steps-.sh
        checkpoint=$folder/$item
        output=$out_dir/$steps
    if [ -f $output ];
        then
        continue
    fi
        cat scripts/translate/val_template.sh > $script
        echo "--language $language --max_len $max_len --checkpoint $checkpoint --output $output --input $input" >> $script
        qsub $script
done

