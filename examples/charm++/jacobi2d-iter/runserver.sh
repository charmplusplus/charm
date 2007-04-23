#!/bin/sh
echo "Starting up server:"
./charmrun +p1 ./jacobi2d 2000 100 ++server ++server-port 1234
echo "Server Exited"

