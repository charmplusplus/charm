#!/usr/local/bin/perl

BEGIN { unshift(@INC, "./blib/arch/auto/mdPerl") }
use mdPerl;

mdInit();
$mype = mdMyPe();
$numpes = mdNumPes();
mdPrintf("Hello World from $mype of $numpes\n");
if($mype == 0){
	for($xpe=1; $xpe < $numpes; $xpe++) {
		mdCall($xpe, "printStats");
	}
}
mdScheduler(-1);
mdExit();

sub printStats
{
	local($pe,$hname,$uptime);
	$pe = mdMyPe();
	$hname = `/bin/hostname`;
	chop $hname;
	$uptime = `/usr/bin/uptime`;
	chop $uptime;
	mdPrintf("$hname: $uptime\n");
	mdExitScheduler();
	mdCall(0, "printStats") if (($pe+1) == mdNumPes());
}

