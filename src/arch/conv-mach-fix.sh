#!/bin/sh
# A small script for removing unneeded symbols
# from all the conv-mach files.  You'll need to edit
# the script to remove the symbols you want to remove.

for f in `echo */conv-mach.h`
do
	echo "Fixing $f..."
	cp $f $f.bak
	cat $f.bak \
	| grep -v 'CMK_USE_HP_MAIN_FIX[ ]*0' \
	| grep -v 'CMK_DONT_USE_HP_MAIN_FIX' \
	> tmp
	mv tmp $f
done
