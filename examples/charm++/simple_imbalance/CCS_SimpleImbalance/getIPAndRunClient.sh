#!/bin/bash

/lustre/scratch/cheelee/development/SimpleCcsClient/simpleCcsClient `grep "Server IP" *.OU | cut -d" " -f5 | cut -d"," -f1` 11337
