#!/bin/bash
let "i = 0";
let "max = 100";
let "numChunks = 2";
let "chunk = 0";
while [ $chunk -le $numChunks ]
do
  grep -i \\[$chunk\\] jnk > jnk.$chunk;
  while [ $i -le $max ]
    do
    export locks=`grep Got\ write\ lock\ on\ node\ $i\{ jnk.$chunk | wc -l`;
    export unlocks=`grep Unlocked\ write\ lock\ on\ node\ $i\{ jnk.$chunk | wc -l`;
    if [ $locks -gt $unlocks ]; then
	echo "$i on Chunk $chunk was acquired $locks times but freed $unlocks times";
    fi
    let "i = i+1";
  done
  let "i = 0";
  let "chunk = chunk+1";
done
