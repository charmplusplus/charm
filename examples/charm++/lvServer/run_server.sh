#!/bin/sh
echo "Starting up server:"
./charmrun ++hierarchical-start +p2 ./lvServer ++server ++server-port 1234  $*
echo "Server exited."
