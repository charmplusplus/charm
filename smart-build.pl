#!/usr/bin/perl


# This is an interactive script that knows
# common ways to build Charm++ and AMPI.
#
# Authors: dooley, becker



# Turn off I/O buffering
$| = 1;

print "Begin interactive charm configuration\n\n";

# Use uname to get the cpu type and OS information
$os = `uname -s`;
$cpu = `uname -m`;

#Variables to hold the portions of the configuration:
$nobs = "";
$arch = "";
$network_option_string = "";
$compiler = "";

#remove newlines from these strings:
chomp($os);
chomp ($cpu);


# Determine OS kernel
if ($os eq "Linux") {
  print "Detected a linux kernel\n";
  $arch_os = "linux";
} elsif ($os eq "Darwin") {
  print "Detected a darwin kernel\n";
  $arch_os = "darwin";
} elsif ($os =~ m/BSD/ ) {
  print "Detected a BSD kernel\n";
  $arch_os = "linux";
} elsif ($os =~ m/OSF1/ ) {
  print "Detected an OSF1 kernel\n";
  $arch_os = "linux";
} elsif ($os =~ m/AIX/ ) {
  print "Detected an AIX kernel\n";
  $arch = "mpi-sp";
}



# Determine architecture (x86, ppc, ...)
if($cpu =~ m/i[0-9]86/){
  print "Detected architecture x86\n";
  $x86 = 1;
} elsif($cpu =~ m/x86\_64/){
  print "Detected architecture x86_64\n";
  $amd64 = 1;
} elsif($cpu =~ m/ia64/){
  print "Detected architecture ia64\n";
  $ia64 = 1;
  $nobs = "--no-build-shared";
} elsif($cpu =~ m/powerpc/){
  print "Detected architecture ppc\n";
  $ppc = 1;
} elsif($cpu =~ m/Power Mac/){
  print "Detected architecture ppc\n";
  $ppc = 1;
} elsif($cpu =~ m/alpha/){
  print "Detected architecture alpha\n";
  $alpha = 1;
}


# Determine converse architecture (net, mpi, ...)
print "Do you have a special network interconnect? [y/N]";
$special_network = "false";
while($line = <>){
	chomp $line;
	if(lc($line) eq "y" || lc($line) eq "yes" ){
		$special_network = "true";
		last;
	} elsif(lc($line) eq "n" || lc($line) eq "no" || $line eq "" ){
		last;
	}
}

# default to net
$converse_network_type = "net";
	
if($special_network eq "true"){
	print << "EOF";
	
Choose an interconnect from below: [1-11]
	 1) MPI
	 2) Infiniband (native ibverbs alpha version)
	 3) Myrinet GM
	 4) Myrinet MX
	 5) Amasso
	 6) Cray XT3, XT4 (not yet tested on CNL)
	 7) Bluegene/L Native (only at T. J. Watson)
	 8) Bluegene/L MPI
	 9) Bluegene/P Native (only at T. J. Watson)
	10) Bluegene/P MPI
	11) VMI

EOF
	
	while($line = <>){
		chomp $line;
		if($line eq "1"){
			$converse_network_type = "mpi";
			last;
		} elsif($line eq "2"){
			$converse_network_type = "net";
			$network_option_string = $network_option_string . "ibverbs ";
			last;
		} elsif($line eq "3"){
			$converse_network_type = "net";
			$network_option_string = $network_option_string . "gm ";
			last;
		} elsif($line eq "4"){
			$converse_network_type = "net";
			$network_option_string = $network_option_string . "mx ";
			last;
		} elsif($line eq "5"){
			$arch = "ammasso";
			last;
		} elsif($line eq "6"){
			$arch = "mpi-crayxt3";
			last;
		} elsif($line eq "7"){
			$arch = "bluegenel";
			$compiler = "xlc";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "8"){
		    $arch = "mpi-bluegenel";
			$compiler = "xlc";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "9"){
		    $arch = "bluegenep";
			$compiler = "xlc";
			last;
		} elsif($line eq "10"){
		    $arch = "mpi-bluegenep";
			$compiler = "xlc";
			last;
		} elsif($line eq "11"){
			$converse_network_type = "mpi";
			$compiler = "vmi";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
	}	
}


if($arch eq ""){
	  $arch = "${converse_network_type}-${arch_os}";
	  if($amd64) {
		$arch = $arch . "-amd64";
	  } elsif($ia64){
	  	$arch = $arch . "-ia64";
	  } elsif($ppc){
	  	$arch = $arch . "-ppc";
	  } elsif($alpha){
		$arch = $arch . "-axp";
	  }
}
  
#Cleanup the architectures to match the horrible real world inconsistent src/archs

if($arch eq "net-darwin"){
	$arch = "net-darwin-x86";
} elsif($arch eq "net-darwin-ppc"){
	$arch = "net-ppc-darwin";
} elsif($arch eq "mpi-darwin-ppc"){
	$arch = "mpi-ppc-darwin";
} 


if($compiler eq "xlc"){
	print "We determined that you should use the compiler $compiler\n Do you want to use a different compiler?[y/N]";
} else {
	print "Do you want to specify a compiler? [y/N]";
}
$special_compiler = "false";
while($line = <>){
	chomp $line;
	if(lc($line) eq "y" || lc($line) eq "yes" ){
		$special_compiler = "true";
		last;
	} elsif(lc($line) eq "n" || lc($line) eq "no" || $line eq ""){
		last;
	} else {
		print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
	}
}




# Choose compiler
if($special_compiler eq "true"){
	#select type of interconnect here
	print << "EOF";
	
Choose a compiler from below: [1-15]

	1) cc
	2) cc64
	3) cxx
	4) kcc
	5) pgcc
	6) acc
	7) icc
	8) ecc
	9) gcc3
	10) gcc4
	11) mpcc
	12) pathscale
	13) xlc
	14) xlc64
    15) mpicxx

EOF

	while($line = <>){
		chomp $line;
		if($line eq "1"){
			$compiler = $compiler . "cc";
			last;
		} elsif($line eq "2"){
			$compiler =  $compiler . "cc64";
			last;
		} elsif($line eq "3"){
			$compiler =  $compiler . "cxx";
			last;
		} elsif($line eq "4"){
			$compiler =  $compiler . "kcc";
			last;
		} elsif($line eq "5"){
			$compiler =  $compiler . "pgcc";
			last;
		} elsif($line eq "6"){
			$compiler =  $compiler . "acc";
			last;
		} elsif($line eq "7"){
			$compiler =  $compiler . "icc";
			last;
		} elsif($line eq "8"){
			$compiler =  $compiler . "ecc";
			last;
		} elsif($line eq "9"){
			$compiler =  $compiler . "gcc3";
			last;
		} elsif($line eq "10"){
			$compiler =  $compiler . "gcc4";
			last;
		} elsif($line eq "11"){
			$compiler =  $compiler . "mpcc";
			last;
		} elsif($line eq "12"){
			$compiler =  $compiler . "pathscale";
			last;
		} elsif($line eq "13"){
			$compiler =  $compiler . "xlc";
			last;
		} elsif($line eq "14"){
			$compiler =  $compiler . "xlc64";
			last;
		} elsif($line eq "15"){
			$compiler =  $compiler . "mpicxx";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
	}
}


# Dynamically generate a list of compilers that could be used


print "Potential compilers:\n---------\n";

@ccs = `find src/arch | grep "cc-"`;
foreach $cc (@ccs) {
  $cc =~ m/cc-([a-zA-Z0-9]*)\..*/;
  print "$1\n";
}

print "\n---------\n";












# Determine whether to support SMP / Multicore
print "Do you want SMP or multicore support? [y/N]";
$smp = "";
while($line = <>){
	chomp $line;
	if(lc($line) eq "y" || lc($line) eq "yes" ){
		$smp = "smp";
		last;
	} elsif(lc($line) eq "n" || lc($line) eq "no" || $line eq ""){
		last;
	} else {
		print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
	}
}


# Choose compiler flags
print << "EOF";
	
Choose a set of compiler flags [1-4]
	1) none
	2) debug
	3) optimized [default]
	4) custom
	
EOF

$compiler_flags = "";

while($line = <>){
	chomp $line;
	if($line eq "1"){
		last;
	} elsif($line eq "2"){
		$compiler_flags = "-g -O0";
		last;
	} elsif($line eq "3" || $line eq ""){
		$compiler_flags = "-O2 -DCMK_OPTIMIZE";
		last;
	} elsif($line eq "4"){

		print "Enter compiler options: ";
		$input_line = <>;
		chomp($input_line);
		$compiler_flags = $input_line;
		
		last;
	} else {
		print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
	}
}




# Determine the target to build.
# We want this simple so we just give 2 options
$target = "";

print << "EOF";

What do you want to build?
	1) Charm++ [default] (choose this if you are building NAMD)
	2) Charm++, AMPI, ParFUM, FEM and other libraries

EOF

while($line = <>){
	chomp $line;
	if($line eq "1" || $line eq ""){
		$target = "charm++";
		last;
	} elsif($line eq "2"){
		$target = "LIBS";
		last;
	} else {
		print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
	}
	
}

# Determine whether to use a -j4 flag for faster building
# Currently LIBS cannot be safely built with -j4
if($target eq "charm++"){
	print "Do you want to do a parallel build (-j4)?[Y/n]";
	while($line = <>){
		chomp $line;
		if(lc($line) eq "y" || lc($line) eq "yes" || $line eq ""){
			$j = "-j4";
		  last;
		} elsif(lc($line) eq "n" || lc($line) eq "no" ){
			$j = "";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
	}
}


# Compose the build line
$build_line = "./build $target $arch ${network_option_string} $compiler $smp $j ${compiler_flags}\n";


# Save the build line in the log
open(BUILDLINE, ">>smart-build.log");
print BUILDLINE `date`;
print BUILDLINE "Using the following build command:\n";
print BUILDLINE "$build_line\n";
close(BUILDLINE);


# Execute the build line if the appriate architecture directory exists
if(-e "src/arch/$arch"){
	print "Building with: ${build_line}\n";	
	# Execute the build line
	system($build_line);
} else {
	print "We could not figure out how to build charm with those options on this platform, please manually build\n";
	print "Try something similar to: ${build_line}\n";
}


