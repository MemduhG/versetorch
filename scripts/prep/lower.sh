turs="data/tr/tur.dev.src data/tr/tur.test.src data/tr/tur.train.src data/tr/prose.txt"
czs="data/cz/cz.dev.src data/cz/cz.test.src data/cz/cz.train.src data/cz/prose.txt"
engs="data/en/eng.dev.src data/en/eng.test.src data/en/eng.train.src data/en/prose.txt"

for item in $turs $czs $engs;
	do
	new_name=`echo $item | sed "s/\([^\.]*\)\./\1-lower\./" `
	echo $item $new_name
	cat $item | awk '{print tolower($0)}' > $new_name
	done

turs="data/tr/tur.dev.tgt data/tr/tur.test.tgt data/tr/tur.train.tgt"
czs="data/cz/cz.dev.tgt data/cz/cz.test.tgt data/cz/cz.train.tgt"
engs="data/en/eng.dev.tgt data/en/eng.test.tgt data/en/eng.train.tgt"

for item in $turs $czs $engs;
	do
	new_name=`echo $item | sed "s/\([^\.]*\)\./\1-lower\./" `
	echo $item $new_name
	cp $item $new_name
	done
