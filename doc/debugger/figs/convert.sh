#!/bin/sh

for basename in arrayelement menu snapshot1 snapshot2 snapshot3 snapshot4
do
	echo "Converting figure $basename"
	convert $basename.png $basename.eps
#	convert $basename.png $basename.pdf
done
