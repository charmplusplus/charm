#!/usr/local/bin/perl

$file = 'words';

foreach $char (ord('a')..ord('z')) {
	$counts[$char] = 0;
}
$start = time;
open(F, $file);
while(<F>) {
	chop;
	@chars = split(//);
	foreach $char (@chars) {
		$counts[ord($char)] ++;
	}
}
$end = time;
close(F);
$elapsed = $end - $start;
foreach $char (ord('a')..ord('z')) {
	print "Frequency of $char is $counts[$char]\n";
}
print "Time taken = $elapsed seconds\n";
exit 0;
