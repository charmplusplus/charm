
In essence, one producer puts many values into a channel, and every consumer scans every channel for a value. Every few iterations, the consumers put a value into a channel and the consumer waits for all such values to be received. The consumers span many chare-arrays. This applies a lot stress to the multi-channel construct, particularly stressing its any-channel receive functionality.

Variables:
    - `NUM_ITERS` - Number of iterations to run for (and channels to create).
    - `numArrays` - Number of chare-arrays to create.
    - `elementsPerArray` - Number of chares per chare-array.
    - `itersPerHeartbeat` - Number of iterations between consumer check-ins.
