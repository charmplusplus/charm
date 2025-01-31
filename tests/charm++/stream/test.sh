#!/bin/bash

SUCCESS_COUNT=0
TOTAL_RUNS=100
EXPECTED_PATTERN_1="Consumer 0 has received the done signal and consumed ([0-9]+) size_t"
EXPECTED_PATTERN_2="Consumer 1 has received the done signal and consumed ([0-9]+) size_t"

for ((i=1; i<=TOTAL_RUNS; i++)); do
    echo "Iteration: $i"
    ./charmrun +p4 ++local ./streamtest > log.out &
    CMD_PID=$!  # Store the process ID of the last background command
    echo "Cmd pid: $CMD_PID"    
    sleep 2
    kill -SIGKILL $CMD_PID  # Forcefully terminate the process
    
    wait $CMD_PID 2>/dev/null  # Wait for the process to terminate
    X=$(awk '/Consumer 0 has received the done signal and consumed/ {print $(NF-1)}' log.out)
    Y=$(awk '/Consumer 1 has received the done signal and consumed/ {print $(NF-1)}' log.out)

    if [[ -n "$X" && -n "$Y" && $((X + Y)) -eq 40 ]]; then
        ((SUCCESS_COUNT++))
    else
        echo "Fail detected on run $i"
        echo "Consumer 0: $X, Consumer 1: $Y"
        break
    fi
done

if [ "$SUCCESS_COUNT" -eq "$TOTAL_RUNS" ]; then
    echo "success"
else
    echo "fail"
fi

