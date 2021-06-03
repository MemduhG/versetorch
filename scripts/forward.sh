machine=`qstat -f $1 | grep "exec_host =" | cut -f2 -d "=" | cut -f1 -d "/"`
echo $machine

