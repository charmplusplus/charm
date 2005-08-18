#!/bin/sh
echo "Starting up server:"
./charmrun +p2 ./lvServer ++server ++server-port 1234  $*
echo "Server exited."
