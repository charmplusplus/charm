# input set of strings to match
# this script removes header files reported from g++ -MM from
# those directories in the argument to this script

@dontprint = @ARGV;
@ARGV=();

$n = @dontprint;

# read from stdin

while (<>) {
# if line ends with : it is the start of a dependency
  chop;
  if ( ($target,$other) = /([a-zA-Z0-9_-]*\.o:)(.*)$/ ) {
    print $target;
    $go=1;
    $first=1;
    while ($go) {
	if ($first) {
	  $_ = $other;
	} else {
	  $_ = <> || last;
	  chop;
	}
	$first = 0;

	if ( /\\$/ ) {
	  chop;
	  $go = 1;
	} else {
	  $go = 0;
	}

	@files = split;
	foreach $word (@files) {
	  $bad = 0;
	  foreach $notword (@dontprint) {
	    if ( $word =~ /$notword/ ) {
	      $bad = 1;
	      last;
	    }
	  }
	  if ( ! $bad ) {
	      print " \\\n";
	      print "	",$word;
	  }
	}
    }
  }
  print "\n";
}

