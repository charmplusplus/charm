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
print "\nBegin interactive charm configuration ...\n";
print "If you are a poweruser expecting a list of options, please use ./build --help\n";
print "\n============================================================\n\n\n";


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
} elsif ($os =~ m/CYGWIN/ ) {
  print "Detected a Cygwin system\n";
  print "This uses the gnu compiler by default,\n";
  print "To build with Microsoft Visual C++ compiler, use net-win32. Please refer to README.win32 for the details on setting up VC++ under cygwin.\n\n";
  $arch_os = "cygwin";
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
} elsif($cpu =~ m/ppc*/){
  $ppc = 1;
} elsif($cpu =~ m/alpha/){
  $alpha = 1;
}



# default to net
$converse_network_type = "net";


# check for MPI

$skip_choosing = "false";

$mpi_found = "false";
$m = system("which mpicc mpiCC > /dev/null 2>/dev/null") / 256;
if($m == 0){
    $mpi_found = "true";
    $mpioption = "";
}
$m = system("which mpicc mpicxx > /dev/null 2>/dev/null") / 256;
if($m == 0){
    $mpi_found = "true";
    $mpioption = "mpicxx";
}

# Give option of just using the mpi version if mpicc and mpiCC are found
if($mpi_found eq "true"){
  print "\nI found that you have an mpicc available in your path.\nDo you want to build Charm++ on this MPI? [y/N]: ";
  $p = promptUserYN();
  if($p eq "yes"){
	$converse_network_type = "mpi";
	$skip_choosing = "true";
	$options = "$options $mpioption";
  }	
}

if($skip_choosing eq "false") { 
  
  print "\nDo you have a special network interconnect? [y/N]: ";
  $p = promptUserYN();
  if($p eq "yes"){
	print << "EOF";
	
Choose an interconnect from below: [1-11]
	 1) MPI
	 2) Infiniband (native ibverbs alpha version)
	 3) Myrinet GM
	 4) Myrinet MX
	 5) LAPI
	 6) Cray XT3, XT4
         7) Cray XT5
	 8) Bluegene/L Native (only at T. J. Watson)
	 9) Bluegene/L MPI
        10) Bluegene/P Native
	11) Bluegene/P MPI
	12) VMI

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
	        $arch = "mpi-crayxt";
	        last;
	  } elsif($line eq "8"){
		$arch = "bluegenel";
		$compilers = "xlc ";
		$nobs = "--no-build-shared";
		last;
	  } elsif($line eq "9"){
		$arch = "mpi-bluegenel";
		$compilers = "xlc ";
		$nobs = "--no-build-shared";
		last;
	  } elsif($line eq "10"){
		$arch = "bluegenep";
		$compilers = "xlc ";
		last;
	  } elsif($line eq "11"){
		$arch = "mpi-bluegenep";
		$compilers = "xlc ";
		last;
	  } elsif($line eq "12"){
		$converse_network_type = "vmi";
		last;
	  } else {
		print "Invalid option, please try again :P\n"
	  }
	}	
  }
}


# construct an $arch string if we did not explicitly set one above
if($arch eq ""){
  $arch = "${converse_network_type}-${arch_os}";
	  if($amd64) {
		$arch = $arch . "-x86_64";
	  } elsif($ia64){
	  	$arch = $arch . "-ia64";
	  } elsif($ppc){
	  	$arch = $arch . "-ppc";
	  } elsif($alpha){
		$arch = $arch . "-axp";
	  }
}
  
# Fixup $arch to match the inconsistent directories in src/archs

if($arch eq "net-darwin"){
	$arch = "net-darwin-x86";
} elsif($arch eq "net-ppc-darwin"){
	$arch = "net-darwin-ppc";
} elsif($arch eq "mpi-ppc-darwin"){
	$arch = "mpi-darwin-ppc";
} elsif($arch eq "multicore-linux-x86_64"){
	$arch = "multicore-linux64";
} 





#================ Choose SMP/PXSHM =================================

# find what options are available
$opts = `./build charm++ $arch help 2>&1 | grep "Supported options"`;
$opts =~ m/Supported options: (.*)/;
$opts = $1;


#always provide multicore and single-threaded versions
print << "EOF";
How do you want to handle SMP/Multicore: [1-4]
         1) single-threaded [default]
         2) multicore(single node only)
EOF

# only add the smp or pxshm options if they are available
$counter = 3; # the next index used in the list

$smpIndex = -1;
if($opts =~ m/smp/){
  print "         $counter) SMP\n";
  $smpIndex = $counter; 
  $counter ++;
}

$pxshmIndex = -1;
if($opts =~ m/pxshm/){
  print "         $counter) POSIX Shared Memory\n";
  $pxshmIndex = $counter; 
  $counter ++;
}

while($line = <>){
	chomp $line;
	if($line eq "1" || $line eq ""){
	    last;
	} elsif($line eq "2"){
	    $converse_network_type = "multicore";
	    last;
	} elsif($line eq $smpIndex){
	    $options = "$options smp ";
	    last;
	} elsif($line eq $pxshmIndex){
	    $options = "$options pxshm ";
	    last;
	}
}







#================ Choose Compiler =================================

# Lookup list of compilers
$cs = `./build charm++ $arch help 2>&1 | grep "Supported compilers"`;
# prune away beginning of the line
$cs =~ m/Supported compilers: (.*)/;
$cs = $1;
# split the line into an array
@c_list = split(" ", $cs);

# print list of compilers
$numc = @c_list;

if ($numc > 0) {
    print "\nDo you want to specify a compiler? [y/N]";
    $p = promptUserYN();
    if($p eq "yes" ){
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
$explanations{"mlogft"} = "Use message logging fault tolerance support";
$explanations{"causalft"} = "Use causal message logging fault tolerance support";





  # Produce list of options

  $opts = `./build charm++ $arch help 2>&1 | grep "Supported options"`;
  # prune away beginning of line
  $opts =~ m/Supported options: (.*)/;
  $opts = $1;

  @option_list = split(" ", $opts);
  

  # Prune out entries that would already have been chosen above, such as smp
  @option_list_pruned = ();
  foreach $o (@option_list){
	if($o ne "smp" && $o ne "ibverbs" && $o ne "gm" && $o ne "mx"){
	  @option_list_pruned = (@option_list_pruned , $o);
	}
  }

  # sort the list
  @option_list_pruned = sort @option_list_pruned;
  if (@option_list_pruned > 0) {

      print "\nDo you want to specify any Charm++ build options, such as fortran compilers? [y/N]";
      $special_options = promptUserYN();

      if($special_options eq "yes"){

          # print out list for user to select from
          print "Please enter one or more numbers separated by spaces\n";
          print "Choices:\n";
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
  }









# Choose compiler flags
print << "EOF";
	
Choose a set of compiler flags [1-5]
	1) none
	2) debug                      -g -O0
	3) optimized [default]        -O2
	4) optimized no projections   -O2 -DCMK_OPTIMIZE
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
	} elsif($line eq "4" ){
 		$compiler_flags = "-O2 -DCMK_OPTIMIZE";
		last;
	} elsif($line eq "3" || $line eq ""){ 
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

# Determine whether to use a -j flag for faster building
$j = "";
    print << "EOF";
    
Do you want to compile in parallel?
        1) No
        2) Build with -j2
        3) Build with -j4
        4) Build with -j8 
        5) Build with -j16 [default]
        6) Build with -j32
        7) Build with -j

EOF

    while($line = <>) {
        chomp $line;
        if($line eq "1"){
	    $j = "";
	    last;
        } elsif($line eq "2") {
	    $j = "-j2";
	    last; 
	} elsif($line eq "3") {
	    $j = "-j4";
	    last;
	}  elsif($line eq "4") {
	    $j = "-j8";
	    last;
	}  elsif($line eq "5" || $line eq "") {
	    $j = "-j16";
	    last;
	}  elsif($line eq "6") {
            $j = "-j32";
            last;
        }  elsif($line eq "7") {
            $j = "-j";
            last;
        }   else {
	    print "Invalid option, please try again :P\n";
	}
}


# Compose the build line
$build_line = "./build $target $arch $compilers $options $smp $j $nobs ${compiler_flags}\n";


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



