#!/bin/sh
echo "Starting up server:"
./charmrun +p 4 ./wave2d +vp 256 ++server ++server-port 1234
echo "Server Exited"

