#!/usr/bin/perl


# This is an interactive script that knows
# common ways to build Charm++ and AMPI.
#
# Authors: dooley, becker


# Turn off I/O buffering
$| = 1;



# A subroutine that reads from input and returns a yes/no/default
sub promptUserYN {
  while($line = <>){
	chomp $line;
	if(lc($line) eq "y" || lc($line) eq "yes" ){
	  return "yes";
	} elsif(lc($line) eq "n" || lc($line) eq "no" ){
	  return "no";
	} elsif( $line eq "" ){
	  return "default";
	}
  }
}
  

# The beginning of the good stuff:
print "\n============================================================\n";
print "\nBegin interactive charm configuration\n";
print "If you are a poweruser expecting a list of options, please use ./build --help\n\n";



# Use uname to get the cpu type and OS information
$os = `uname -s`;
$cpu = `uname -m`;

#Variables to hold the portions of the configuration:
$nobs = "";
$arch = "";
$compilers = "";
$options = "";

#remove newlines from these strings:
chomp($os);
chomp ($cpu);


# Determine OS kernel
if ($os eq "Linux") {
  $arch_os = "linux";
} elsif ($os eq "Darwin") {
  $arch_os = "darwin";
} elsif ($os =~ m/BSD/ ) {
  $arch_os = "linux";
} elsif ($os =~ m/OSF1/ ) {
  $arch_os = "linux";
} elsif ($os =~ m/AIX/ ) {
  $arch = "mpi-sp";
}



# Determine architecture (x86, ppc, ...)
if($cpu =~ m/i[0-9]86/){
  $x86 = 1;
} elsif($cpu =~ m/x86\_64/){
  $amd64 = 1;
} elsif($cpu =~ m/ia64/){
  $ia64 = 1;
  $nobs = "--no-build-shared";
} elsif($cpu =~ m/powerpc/){
  $ppc = 1;
} elsif($cpu =~ m/Power Mac/){
  $ppc = 1;
} elsif($cpu =~ m/alpha/){
  $alpha = 1;
}



# Determine converse architecture (net, mpi, ...)

# default to net
$converse_network_type = "net";

print "Do you have a special network interconnect? [y/N]";
$p = promptUserYN();
if($p eq "yes"){
	print << "EOF";
	
Choose an interconnect from below: [1-11]
	 1) MPI
	 2) Infiniband (native ibverbs alpha version)
	 3) Myrinet GM
	 4) Myrinet MX
	 5) LAPI
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
			$options = "$options ibverbs ";
			last;
		} elsif($line eq "3"){
			$converse_network_type = "net";
			$options = $options . "gm ";
			last;
		} elsif($line eq "4"){
			$converse_network_type = "net";
			$options = $options . "mx ";
			last;
		} elsif($line eq "5"){
			$arch = "lapi";
			last;
		} elsif($line eq "6"){
			$arch = "mpi-crayxt3";
			last;
		} elsif($line eq "7"){
			$arch = "bluegenel";
			$compilers = "xlc ";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "8"){
		    $arch = "mpi-bluegenel";
			$compilers = "xlc ";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "9"){
		    $arch = "bluegenep";
			$compilers = "xlc ";
			last;
		} elsif($line eq "10"){
		    $arch = "mpi-bluegenep";
			$compilers = "xlc ";
			last;
		  } elsif($line eq "11"){
			$converse_network_type = "vmi";
			last;
		} else {
			print "Invalid option, please try again :P\n"
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
  
# Fixup the architecture to match the horrible real 
# world inconsistent directories in src/archs

if($arch eq "net-darwin"){
	$arch = "net-darwin-x86";
} elsif($arch eq "net-darwin-ppc"){
	$arch = "net-ppc-darwin";
} elsif($arch eq "mpi-darwin-ppc"){
	$arch = "mpi-ppc-darwin";
} 




# Determine whether to support SMP / Multicore
print "Do you want SMP or multicore support? [y/N]";
$p = promptUserYN();
if($p eq "yes" ){
  $options = "$options smp ";
}


#================ Choose Compiler =================================


print "Do you want to specify a compiler? [y/N]";
$p = promptUserYN();
if($p eq "yes" ){

  # Lookup list of compilers
  $cs = `./build charm++ $arch help | grep "Supported compilers"`;
  # prune away beginning of the line
  $cs =~ m/Supported compilers: (.*)/;
  $cs = $1;
  # split the line into an array
  @c_list = split(" ", $cs);

  # print list of compilers
  $numc = @c_list;
  print "Choose a compiler: [1-$numc] \n";

  $i = 1;
  foreach $c (@c_list){
	print "\t$i)\t$c\n";
	$i++;
  }
  
  # Choose compiler
  while($line = <>){
	chomp $line;
	if($line =~ m/([0-9]*)/ && $1 > 0 && $1 <= $numc){
	  $compilers = $c_list[$1-1];
	  last;
	} else {
	  print "Invalid option, please try again :P\n"
	}
  }
}




#================ Choose Options =================================

#Create a hash table containing descriptions of various options
%explanations = ();
$explanations{"ooc"} = "Out-of-core execution support in Charm++";
$explanations{"tcp"} = "Charm++ over TCP instead of UDP for net versions. TCP is slower";
$explanations{"ifort"} = "Use Intel's ifort fortran compiler";
$explanations{"gfortran"} = "Use gfortran compiler for fortran";
$explanations{"g95"} = "Use g95 compiler";
$explanations{"ifort"} = "Use Intel's ifort fortran compiler";
$explanations{"pgf90"} = "Use Portland Group's pgf90 fortran compiler";
$explanations{"ifc"} = "Use Intel's ifc compiler";
$explanations{"ammasso"} = "Use native RDMA support on Ammasso interconnect";
$explanations{"syncft"} = "Use initial fault tolerance support";



print "Do you want to specify any Charm++ build options such as fortran compilers? [y/N]";
$special_options = promptUserYN();

if($special_options eq "yes"){

  # Produce list of options

  $opts = `./build charm++ $arch help | grep "Supported options"`;
  # prune away beginning of line
  $opts =~ m/Supported options: (.*)/;
  $opts = $1;

  @option_list = split(" ", $opts);
  
  print "Please enter one or more numbers separated by spaces\n";
  print "Choices:\n";

  # Prune out entries that would already have been chosen above, such as smp
  @option_list_pruned = ();
  foreach $o (@option_list){
	if($o ne "smp" && $o ne "ibverbs" && $o ne "gm" && $o ne "mx"){
	  @option_list_pruned = (@option_list_pruned , $o);
	}
  }

  # sort the list
  @option_list_pruned = sort @option_list_pruned;

  # print out list for user to select from
  $i = 1;
  foreach $o (@option_list_pruned){
	$exp = $explanations{$o};
	print "\t$i)\t$o";
	# pad whitespace before options
	for($j=0;$j<20-length($o);$j++){
	  print " ";
	}
    print ": $exp";
	print "\n";
	$i++;
  }
  print "\t$i)\tNone Of The Above\n";

  $num_options = @option_list_pruned;

  while($line = <>){
	chomp $line;
    $line =~ m/([0-9 ]*)/;
    @entries = split(" ",$1);
    @entries = sort(@entries);

	$additional_options = "";
	foreach $e (@entries) {
	  if($e>=1 && $e<= $num_options){
		$estring = $option_list_pruned[$e-1];
		$additional_options = "$additional_options $estring";
	  } elsif ($e == $num_options+1){
		# user chose "None of the above"
		# clear the options we may have seen before
		$additional_options = " ";
	  }
	}

	# if the user input something reasonable, we can break out of this loop
	if($additional_options ne ""){
	  $options = "$options ${additional_options} ";
	  last;
	}

  }

}




# Choose compiler flags
print << "EOF";
	
Choose a set of compiler flags [1-5]
	1) none
	2) debug
	3) optimized [default]
	4) optimized + projections
	5) custom
	
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
                $compiler_flags = "-O2"; 
                last; 
        }  elsif($line eq "5"){

		print "Enter compiler options: ";
		$input_line = <>;
		chomp($input_line);
		$compiler_flags = $input_line;
		
		last;
	} else {
		print "Invalid option, please try again :P\n"
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
		print "Invalid option, please try again :P\n"
	}
	
}

# Determine whether to use a -j4 flag for faster building
# Currently LIBS cannot be safely built with -j4
$j = "";
if($target eq "charm++"){
	print "Do you want to do a parallel build (-j4)?[Y/n]";
	$p = promptUserYN();
	if($p eq "yes" || $p eq "default"){
	  $j = "-j4";
	}
}


# Compose the build line
$build_line = "./build $target $arch $compilers $options $smp $j ${compiler_flags}\n";


# Save the build line in the log
open(BUILDLINE, ">>smart-build.log");
print BUILDLINE `date`;
print BUILDLINE "Using the following build command:\n";
print BUILDLINE "$build_line\n";
close(BUILDLINE);


print "We have determined a suitable build line is:\n";
print "\t$build_line\n\n";


# Execute the build line if the appropriate architecture directory exists
print "Do you want to start the build now? [Y/n]";
$p = promptUserYN();
if($p eq "yes" || $p eq "default"){
  if(-e "src/arch/$arch"){
	print "Building with: ${build_line}\n";	
	# Execute the build line
	system($build_line);
  } else {
	print "We could not figure out how to build charm with those options on this platform, please manually build\n";
	print "Try something similar to: ${build_line}\n";
  }
}



