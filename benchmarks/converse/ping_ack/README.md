# Ping_ack Benchmark

This benchmark is a test of fine-grained messages between two processes on two different nodes, specifically focusing on the injection rate of repeated small messages into the network (as in the LogP model).  
This benchmark tests messages of size 56, 4096, and 65536 bytes over a set of 10 trials of 100 message sends each. You can use the `-msg_count` option to control the message count in each trial.  

## Running the benchmark  
`make`
`<charmrun/srun> -n 2 ./ping_ack +p<pes per process>`

## Results format  
`{#PEs},{msg_size},{average process_time},{avg send_time},{avg total_time},{stdev process_time},{stdev send_time},{stdev total_time},{max process_time},{max send_time},{max total_time}`  
i.e.  
`DATA,2,56,634.696600,11.718200,675.497000,80.270260,0.042251,80.125143,846.293000,11.709000,886.553000`