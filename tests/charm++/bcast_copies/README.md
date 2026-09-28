# bcast_copies: how many copies does a broadcast make?

`nokeepaudit` broadcasts to a group (one branch per PE) and to a 1D chare
array (`perPe` elements per PE) with every kind of entry method -- marshalled,
fixed-size message, varsize message, custom pack/unpack message -- each with
and without `[nokeep]`, plus `[threaded]` variants. Every receiver reports the
buffer it was handed; the program counts the distinct buffers system-wide and
checks each row against the runtime's promise: a nokeep entry sees **one
buffer per process**, a keep entry sees **one per receiver**, and every
payload is intact. A stress phase then sends hundreds of keep broadcasts
back to back while every element migrates, so broadcasts are also delivered
from the array broadcaster's stored copies.

Buffers are told apart by (process, address, marker id): the first receiver
stamps a marker word inside the payload with a job-wide unique id, so a copy
that the allocator placed at a recycled address is still counted (address
reuse was common in the threaded rows), and a copy made after the stamp is
separated by its address.

Not part of the CI tier: it is a hand-run audit and stress program. It found
two heap bugs (charm #4000/#4001: `[threaded, nokeep]` freed a shared
message; #4002/#4003: custom-packed array broadcasts to keep entries reused a
freed source) and a reconverse exit-path bug (reconverse #256).

    ./nokeepaudit [perPe=10] [payloadWords=1024] [verbose=0] [withUnsafe=0]
                  [stressBcasts=200] [stressVarOnly=0]

`withUnsafe` is a bitmask enabling the cases that crashed before the fixes
above: 1 = `[threaded, nokeep]` marshalled, 2 = custom-packed array
broadcast to a keep entry, 4 = threaded-nokeep entries that suspend before
reading. Exit status is 0 only if every row and the stress phase pass. Sites:
run with 4 processes per node, e.g. reconverse
`srun -N 2 --ntasks-per-node=4 -c 8 ./nokeepaudit 10 1024 0 7 400 +pe 64`,
classic `... +ppn 7`.
