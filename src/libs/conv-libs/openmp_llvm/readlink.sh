#!/bin/sh
# For enviroments where GNU readlink is not available in default (e.g. Darwin), this script returns the absolute path of the input directory.
CURRENT_DIR=`pwd`
cd $1
TARGET_DIR=`pwd`
cd $CURRENT_DIR
echo $TARGET_DIR
