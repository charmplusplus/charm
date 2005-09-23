#!/bin/bash

let "mini = 0";
let "i = $mini";
let "maxi = 2";
let "minj = 0";
let "j = $minj";
let "maxj = 2";
let "mink = 0";
let "k = $mink";
let "maxk = 2";
let "count = 1";
let "total = ($maxi+1) * ($maxj+1) * ($maxk+1)";
echo $total;
while [ $i -le $maxi ]
  do
  let "j = 0";
  ycord=$(echo "$i * 0.01" | bc);
  while [ $j -le $maxj ]
    do
    xcord=$(echo "$j * 0.01" | bc);
    let "k = 0";
    while [ $k -le $maxk ]
      do
      zcord=$(echo "$k * 0.01" | bc);
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

      echo $count $xcord $ycord $zcord $bound;
      let "count = count + 1";
      let "k = $k + 1";
    done
    let "j = $j + 1";
  done
  let "i = $i + 1";
done
