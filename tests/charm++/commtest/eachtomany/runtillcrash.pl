#!/usr/bin/perl

$| = 1;

$i = 0;
while(1){
  $i++;
  print "starting run $i\n";
  `time ./charmrun  ./eachtomany +p5 > log`;
  print "\n";
}
