#!/bin/sh
# Runs the command given as arguments and prints its exit status, so that a
# test can check the status of every process a launcher starts.
"$@"
rc=$?
echo "process exit status $rc"
exit $rc
