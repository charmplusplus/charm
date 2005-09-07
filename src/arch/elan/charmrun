#!/bin/sh
#
# Conv-host for HP/Compaq prun-style Supercomputers:
#  Translates +pN-style charmrun options into prun options

args=""
pes=1

while [ $# -gt 0 ]
do
	case $1 in
	+p)
		pes=$2
		shift
		;;
	+p*)
		pes=`echo $1 | awk '{print substr($1,3)}'`
		;;
	*) 
		args=$args"$1 "
		;;
	esac
	shift
done

# Try to guess the number of nodes
nodes=$pes
ppn=0
if [ ! "$RMS_NODES" = "" ] 
then
  ppn=`expr $RMS_PROCS / $RMS_NODES`
  test $pes -gt $RMS_NODES && nodes=$RMS_NODES
fi
for i in 4 2 3
do
  [ $ppn -ne 0 -a $i -gt $ppn ] && continue
  if [ `expr $pes / $i '*' $i` -eq $pes ]
  then
	nodes=`expr $pes / $i`
	break
  fi
done

extra="-N $nodes -n $pes "

# Prepend path to executable
args=`pwd`/"$args"

if [ ! "$RMS_NODES" = "" ]
then
# Running from a batch script: just use prun
	if test $pes -gt $RMS_PROCS
        then
	  echo "Charmrun> too many processors requested!"
        fi
	echo "Charmrun running> prun $extra $args"
	prun $extra $args	
else
# Interactive mode: create, and submit a batch job
	script="charmrun_script.$$.sh"
	indir=`pwd`
	output="./charmrun_script.$$.stdout"
	echo "Submitting batch job for> prun $extra $args"
	echo " using the command> qsub $script"
	cat > $script << EOF
#!/bin/sh
# This is a charmrun-generated PBS batch job script.
# The lines starting with #PBS are queuing system flags:

# This determines the number of nodes and pes (here $nodes and $pes):
#PBS -l rmsnodes=$nodes:$pes

# This determines the wall-clock time limit (here 5 minutes):
#PBS -l walltime=10:00

# This specifies we don't want e-mail info. on this job:
#PBS -m n

# This combines stdout and stderr into one file:
#PBS -j oe

# This specifies the file to write stdout information to:
#PBS -o $output

# Change to the directory where charmrun was run:
cd $indir

# This is the actual command to run the job:
prun $extra $args > $output 2>&1
EOF
	chmod 755 $script
	jobid=`qsub $script`
	echo "Job enqueued under job ID $jobid"

End() {
        echo "autobuild> qdel $jobid ..."
        qdel $jobid
        rm -f $script
        exit $1
}

	trap 'End 1' 2 3
	retry=0
# Wait for the job to complete, by checking its status
	while [ true ]
	do
		qstat $jobid > tmp.$$
		exitstatus=$?
		if [ -f $output ]
		then
# The job is done-- print its output			
			rm tmp.$$ $script
			if `grep 'End of program' $output > /dev/null 2>&1`
                        then
                                exec cat $output
                        else
                                cat $output
                                rm $output
                                exit 1
                        fi
		fi
# The job is still queued or running-- print status and wait
		tail -1 tmp.$$
		rm tmp.$$
		if test $exitstatus -ne 0
                then
# retry a few times when error occurs
                        retry=`expr $retry + 1`
                        if test $retry -gt 6
                        then
                                echo "Charmrun> too many errors, abort!"
                                exit 1
                        else
                                sleep 15
                        fi
                else
# job still in queue
                        retry=0
                        sleep 20
                fi
	done
fi
