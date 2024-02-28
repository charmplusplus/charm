#!/bin/bash

export script=$1;
queue_qsub=sbatch
queue_kill=scancel
queue_stat="squeue -j"

End() {
        echo "autobuild> $queue_kill $jobid ..."
        $queue_kill $jobid
        exit $1
}

echo "Submitting batch job for>  $target OPTS=$flags"
echo " using the command> $queue_qsub $script"
chmod 755 $script

while [ -z "$jobid" ]
do
    $queue_qsub $script > .status.$$ 2>&1
    if grep 'have no accounts' .status.$$ > /dev/null
    then
        echo "NO account for submitting batch job!"
        rm -f .status.$$
        exit 1
    fi
    jobid=`cat .status.$$ | tail -1 | awk '{print $4}'`
    rm -f .status.$$
done

echo "Job enqueued under job ID $jobid"

export output=$jobid.output
export result=$jobid.result

# kill job if interrupted
trap 'End 1' 2 3
retry=0

# Wait for the job to complete, by checking its status
while [ true ]
do
    $queue_stat $jobid > tmp.$$
    linecount=`wc -l tmp.$$ | awk '{print $1}' `
    exitstatus=$?
    #if [[ $exitstatus != 0 || $linecount != 2 ]]
    if [[ $linecount != 2 ]]
    then
        # The job is done-- print its output
        rm tmp.$$
        # When job hangs, result file does not exist
        test -f $result && status=`cat $result` || status=1
        echo "==================================== OUTPUT (STDOUT & STDERR) ========================================"
        cat $output
        echo "======================================================================================================"
        if [[ "$status" != 0 ]];
        then
            #print script
            echo "=============================================== SCRIPT ==============================================="
            cat $script
            echo "======================================================================================================"
            echo "=============================================== RESULT ==============================================="
            cat $result
            echo "======================================================================================================"
        fi

        # mv result and output to result.latest
        mv $result result.latest
        mv $output output.latest

        exit $status
    fi

    # The job is still queued or running-- print status and wait
    tail -1 tmp.$$
    rm tmp.$$
    sleep 60
done
