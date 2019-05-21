This example tests quiescence detection. There is a 1D chare array that simulates
work by busy waiting a given amount of time on the GPU. The Main chare utilizes
quiescence detection to check if all chares have completed and moves on to the
next iteration.

Usage: ./qdtest -c [chares] -i [iterations] -t [busy time] -d [data size]
       -n flag turns off QD and resorts to reduction instead. This can be useful
       to see if QD is working as expected.
