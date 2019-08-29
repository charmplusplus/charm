This example passes a Hello message along the elements of a chare array.
Each chare executes an empty kernel on the GPU when it receives the message.
When the kernel completes, the runtime system executes the specified callback
function which passes the message to the subsequent chare in the array.

Usage: ./hello -c [chares]
