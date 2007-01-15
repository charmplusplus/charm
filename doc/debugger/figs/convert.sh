#!/bin/sh

for name in *.png
do
	basename=${name%.png}
	echo "Converting figure $basename"
	convert $basename.png $basename.eps
#	convert $basename.png $basename.pdf
done
