#!/usr/local/bin/perl

BEGIN { unshift(@INC, "./blib/arch/auto/mdPerl") }
use mdPerl;

$file = 'words';
$size = 206662;
$ncollect = 0;

mdInit();
$mype = mdMyPe();
$numpes = mdNumPes();
$length = int($size/$numpes);
$offset = $mype * $length;

foreach $char (ord('a')..ord('z')) {
	$counts[$char] = 0;
	$gcounts[$char] = 0;
}
if($mype == 0) {
	foreach $pe (0..$numpes-1) {
		mdCall($pe, "count");
	}
}
$start = mdTimer();
mdScheduler(-1);
$end = mdTimer();
$elapsed = $end - $start;
if ($mype == 0) {
	foreach $char (ord('a')..ord('z')) {
		mdPrintf("Frequency of $char is $gcounts[$char]\n");
	}
	mdPrintf("Time taken = $elapsed seconds\n");
}
mdExit();
exit 0;


sub count {
	local($len) = 0;
	open(F, $file);
	seek(F, $offset,0);
	while(<F>) {
		chop; $len++;
		@chars = split(//);
		foreach $char (@chars) {
			$len ++;
			last if ($len >= $length);
			$counts[ord($char)] ++;
		}
		last if ($len >= $length);
	}
	close(F);
	$x = join(' ',@counts[ord('a')..ord('z')]);
	mdCall(0, "collect", $x);
	mdExitScheduler() if($mype != 0);
}

sub collect {
	local(@lcounts);
	@lcounts[ord('a')..ord('z')] = split(' ', pop(@_));
	$ncollect++;
	foreach $char (ord('a')..ord('z')) {
		$gcounts[$char] += $lcounts[$char];
	}
	mdExitScheduler() if($ncollect == $numpes);
}
