#!/usr/bin/perl

$i = 0;
$count_success = 0;
$count_abort = 0;
$runs = 50;

while($i < $runs) {
  $p = 1+$i*5 % 17;
  
  $i++;
  
  $program = "./charmrun ./test +p$p ++local";
  
  print "Running \"$program\"\n";
  $output = `$program`;
  
  if ($output =~ m/Test Completed Successfully/gi) {
    $count_success ++;  
  } 
  
  if($output =~ m/abort/gi){
    $count_abort ++;
    print $output;
  }
  
}

print "${count_success} / $runs Tests succeeded (${count_abort} aborted)\n";


