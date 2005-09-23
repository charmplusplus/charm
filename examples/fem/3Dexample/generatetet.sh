#!/bin/bash

let "i = 0";
let "maxi = 2";
let "j = 0";
let "maxj = 2";
let "k = 0";
let "maxk = 2";
let "count = 1";
let "total = $maxi * $maxj * $maxk * 6";
echo $total;
while [ $i -lt $maxi ]
  do
  let "j = 0";
  while [ $j -lt $maxj ]
    do
    let "k = 0";
    while [ $k -lt $maxk ]
      do
      let "vert1 = $i * ($maxk+1) * ($maxj+1) + $j * ($maxk+1) + $k";
      let "vert2 = $vert1 + 1";
      let "vert3 = $vert1 + ($maxk+1)";
      let "vert4 = $vert3 + 1";
      let "vert5 = $vert1 + ($maxk+1) * ($maxj+1)";
      let "vert6 = $vert5 + 1";
      let "vert7 = $vert5 + ($maxk+1)";
      let "vert8 = $vert7 + 1";
      echo $count $vert1 $vert2 $vert4 $vert6;
      let "count = count + 1";
      echo $count $vert5 $vert8 $vert6 $vert1;
      let "count = count + 1";
      echo $count $vert8 $vert6 $vert4 $vert1;
      let "count = count + 1";
      echo $count $vert1 $vert3 $vert4 $vert7;
      let "count = count + 1";
      echo $count $vert5 $vert8 $vert7 $vert4;
      let "count = count + 1";
      echo $count $vert7 $vert4 $vert1 $vert5;
      let "count = count + 1";
      let "k = $k + 1";
    done
      let "j = $j + 1";
  done
  let "i = $i + 1";
done

