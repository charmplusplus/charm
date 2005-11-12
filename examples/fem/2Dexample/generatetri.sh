#!/bin/bash

let "i = 0";
let "maxi = 999";
let "j = 1";
let "maxj = 1000";
let "count = 1";
let "total = $maxi * ($maxj-1) *2";
echo $total;
while [ $i -lt $maxi ]
  do
  let "j = 1";
  while [ $j -lt $maxj ]
    do
    let "vert1 = $i * $maxj + j";
    let "vert2 = $vert1 + 1";
    let "vert3 = $vert1 + $maxj";
    let "vert4 = $vert3 + 1";
    echo $count $vert1 $vert2 $vert4;
    let "count = count + 1";
    echo $count $vert1 $vert3 $vert4;
    let "count = count + 1";
    let "j = $j + 1";
  done
  let "i = $i + 1";
done

