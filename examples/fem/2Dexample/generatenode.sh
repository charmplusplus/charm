#!/bin/bash

let "mini = 0";
let "i = $mini";
let "maxi = 999";
let "minj = 0";
let "j = $minj";
let "maxj = 999";
let "count = 1";
let "total = ($maxi+1) * ($maxj+1)";
echo $total;
while [ $i -le $maxi ]
  do
  let "j = 0";
  ycord=$(echo "$i * 0.001" | bc);
  let "bound = 0";
  while [ $j -le $maxj ]
    do
    let "bound = 0";
    if [ $i -eq $mini ] 
	then
	let "bound = -1";
    fi
    if [ $j -eq $maxj ]
	then
	let "bound = -2";
    fi
    if [ $i -eq $maxi ]
	then
	let "bound = -3";
    fi
    if [ $j -eq $minj ] 
	then
	let "bound = -4";
    fi
    xcord=$(echo "$j * 0.001" | bc);
    echo $count $xcord $ycord $bound;
    let "count = count + 1";
    let "j = $j + 1";
  done
  let "i = $i + 1";
done
