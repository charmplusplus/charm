#!/usr/local/bin/perl

BEGIN { unshift(@INC, "./blib/arch/auto/mdPerl") }
use mdPerl;

mdInit();
$mype = mdMyPe();
$numpes = mdNumPes();
mdPrintf("Hello World from $mype of $numpes\n");
if($mype == 0){
	for($xpe=1; $xpe < $numpes; $xpe++) {
		mdCall($xpe, "printArgs", pack("a4a4i", "aaa", "bbb", 234));
	}
}
mdScheduler(-1);
mdExit();

sub printArgs
{
	local($args) = @_;
	local($arg1, $arg2, $arg3) = unpack("A4A4i", $args);
	local($pe) = mdMyPe();
	mdPrintf("$pe: arg1=$arg1 arg2=$arg2 arg3=$arg3\n");
	mdExitScheduler();
	mdCall(0, "printArgs", pack("a4a4i", "ccc", "ddd", 4567)) 
		if (($pe+1) == mdNumPes());
}

