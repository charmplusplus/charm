#!/bin/sh

c="convert -density 144x144 "
for fig in conn_indexing conn_indexing_old
do
	echo "Convert $fig.eps $fig.png"
	$c $fig.eps $fig.png || exit 1
done

