#!/usr/bin/perl


# This is an interactive script that knows
# common ways to build Charm++ and AMPI.
#
# Authors: dooley, becker

use strict;
use warnings;

# Turn off I/O buffering
$| = 1;



# A subroutine that reads from input and returns a yes/no/default
sub promptUserYN {
  while(my $line = <>){
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
my $os = `uname -s`;
my $cpu = `uname -m`;

#Variables to hold the portions of the configuration:
my $nobs = "";
my $arch = "";
my $compilers = "";
my $options = "";

#remove newlines from these strings:
chomp($os);
chomp ($cpu);

my $arch_os;
# Determine OS kernel
if ($os eq "Linux") {
  $arch_os = "linux";
} elsif ($os eq "Darwin") {
  $arch_os = "darwin";
} elsif ($os =~ m/BSD/ ) {
  $arch_os = "linux";
} elsif ($os =~ m/OSF1/ ) {
  $arch_os = "linux";
} elsif ($os =~ m/CYGWIN/ ) {
  print "Detected a Cygwin system\n";
  print "This uses the gnu compiler by default,\n";
  print "To build with Microsoft Visual C++ compiler, use net-win32. Please refer to README.win32 for the details on setting up VC++ under cygwin.\n\n";
  $arch_os = "cygwin";
}


my $x86;
my $amd64;
my $ppc;
my $arm7;
# Determine architecture (x86, ppc, ...)
if($cpu =~ m/i[0-9]86/){
  $x86 = 1;
} elsif($cpu =~ m/x86\_64/){
  $amd64 = 1;
} elsif($cpu =~ m/powerpc/){
  $ppc = 1;
} elsif($cpu =~ m/ppc*/){
  $ppc = 1;
} elsif($cpu =~ m/arm7/){
  $arm7 = 1;
}


# default to netlrts
my $converse_network_type = "netlrts";
my $skip_choosing = "false";

print "Are you building to run just on the local machine, and not across multiple nodes? [";
if($arch_os eq "darwin") {
    print "Y/n]\n";
} else {
    print "y/N]\n";
}
{
    my $p = promptUserYN();
    if($p eq "yes" || ($arch_os eq "darwin" && $p eq "default")){
	$converse_network_type = "multicore";
	$skip_choosing = "true";
    }
}

# check for MPI

my $mpi_found = "false";
my $m = system("which mpicc mpiCC > /dev/null 2>/dev/null") / 256;
my $mpioption;
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
if($skip_choosing eq "false" && $mpi_found eq "true"){
  print "\nI found that you have an mpicc available in your path.\nDo you want to build Charm++ on this MPI? [y/N]: ";
  my $p = promptUserYN();
  if($p eq "yes"){
	$converse_network_type = "mpi";
	$skip_choosing = "true";
	$options = "$options $mpioption";
  }	
}

if($skip_choosing eq "false") { 
  
  print "\nDo you have a special network interconnect? [y/N]: ";
  my $p = promptUserYN();
  if($p eq "yes"){
	print << "EOF";
	
Choose an interconnect from below: [1-10]
	 1) MPI
	 2) Infiniband (ibverbs)
	 3) Cray XE, XK
	 4) Cray XC
	 5) Blue Gene/Q

EOF
	
	while(my $line = <>){
	  chomp $line;
	  if($line eq "1"){
		$converse_network_type = "mpi";
		last;
	  } elsif($line eq "2"){
		$converse_network_type = "verbs";
		last;
	  } elsif($line eq "3"){
	        $arch = "gni-crayxe";
	        last;
	  } elsif($line eq "4"){
	        $arch = "gni-crayxc";
	        last;
	  } elsif($line eq "5"){
		$arch = "pamilrts-bluegeneq";
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
	  } elsif($ppc){
	  	$arch = $arch . "-ppc";
	  } elsif($arm7){
	  	$arch = $arch . "-arm7";
	  }
}
  
# Fixup $arch to match the inconsistent directories in src/archs

if($arch eq "netlrts-darwin"){
	$arch = "netlrts-darwin-x86_64";
} elsif($arch eq "multicore-linux-arm7"){
	$arch = "multicore-arm7";
} elsif($arch eq "multicore-linux-x86_64"){
	$arch = "multicore-linux64";
} 


#================ Choose SMP/PXSHM =================================

# find what options are available
my $opts = `./build charm++ $arch help 2>&1 | grep "Supported options"`;
$opts =~ m/Supported options: (.*)/;
$opts = $1;

my $smp_opts = <<EOF;
      1) single-threaded [default]
EOF

# only add the smp or pxshm options if they are available
my $counter = 1; # the last index used in the list

my $smpIndex = -1;
if($opts =~ m/smp/){
  $counter ++;
  $smp_opts = $smp_opts . "      $counter) SMP\n";
  $smpIndex = $counter;
}

my $pxshmIndex = -1;
if($opts =~ m/pxshm/){
  $counter ++;
  $smp_opts = $smp_opts . "      $counter) POSIX Shared Memory\n";
  $pxshmIndex = $counter;
}

if ($counter != 1) {
    print "How do you want to handle SMP/Multicore: [1-$counter]\n";
    print $smp_opts;

    while(my $line = <>){
	chomp $line;
	if($line eq "" || $line eq "1"){
	    last;
	} elsif($line eq $smpIndex){
	    $options = "$options smp ";
	    last;
	} elsif($line eq $pxshmIndex){
	    $options = "$options pxshm ";
	    last;
	}
    }
}


#================ Choose Compiler =================================

# Lookup list of compilers
my $cs = `./build charm++ $arch help 2>&1 | grep "Supported compilers"`;
# prune away beginning of the line
$cs =~ m/Supported compilers: (.*)/;
$cs = $1;
# split the line into an array
my @c_list = split(" ", $cs);

# print list of compilers
my $numc = @c_list;

if ($numc > 0) {
    print "\nDo you want to specify a compiler? [y/N]";
    my $p = promptUserYN();
    if($p eq "yes" ){
        print "Choose a compiler: [1-$numc] \n";

        my $i = 1;
        foreach my $c (@c_list){
            print "\t$i)\t$c\n";
            $i++;
        }

        # Choose compiler
        while(my $line = <>){
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
my %explanations = ();
$explanations{"ooc"} = "Out-of-core execution support in Charm++";
$explanations{"tcp"} = "Charm++ over TCP instead of UDP for net versions. TCP is slower";
$explanations{"gfortran"} = "Use gfortran compiler for fortran";
$explanations{"ifort"} = "Use Intel's ifort fortran compiler";
$explanations{"pgf90"} = "Use Portland Group's pgf90 fortran compiler";
$explanations{"syncft"} = "Use fault tolerance support";
$explanations{"mlogft"} = "Use message logging fault tolerance support";
$explanations{"causalft"} = "Use causal message logging fault tolerance support";





  # Produce list of options

  $opts = `./build charm++ $arch help 2>&1 | grep "Supported options"`;
  # prune away beginning of line
  $opts =~ m/Supported options: (.*)/;
  $opts = $1;

  my @option_list = split(" ", $opts);
  

  # Prune out entries that would already have been chosen above, such as smp
  my @option_list_pruned = ();
  foreach my $o (@option_list){
	if($o ne "smp" && $o ne "ibverbs" && $o ne "gm" && $o ne "mx"){
	  @option_list_pruned = (@option_list_pruned , $o);
	}
  }

  # sort the list
  @option_list_pruned = sort @option_list_pruned;
  if (@option_list_pruned > 0) {

      print "\nDo you want to specify any Charm++ build options, such as fortran compilers? [y/N]";
      my $special_options = promptUserYN();

      if($special_options eq "yes"){

          # print out list for user to select from
          print "Please enter one or more numbers separated by spaces\n";
          print "Choices:\n";
          my $i = 1;
          foreach my $o (@option_list_pruned){
              my $exp = $explanations{$o};
              print "\t$i)\t$o";
              # pad whitespace before options
              for(my $j=0;$j<20-length($o);$j++){
                  print " ";
              }
              print ": $exp";
              print "\n";
              $i++;
          }
          print "\t$i)\tNone Of The Above\n";

          my $num_options = @option_list_pruned;

          while(my $line = <>){
              chomp $line;
              $line =~ m/([0-9 ]*)/;
              my @entries = split(" ",$1);
              @entries = sort(@entries);

              my $additional_options = "";
              foreach my $e (@entries) {
                  if($e>=1 && $e<= $num_options){
                      my $estring = $option_list_pruned[$e-1];
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
	2) debug mode                        -g -O0
	3) production build [default]        --with-production
	4) production build w/ projections   --with-production --enable-tracing
	5) custom
	
EOF

my $compiler_flags = "";

while(my $line = <>){
	chomp $line;
	if($line eq "1"){
		last;
	} elsif($line eq "2"){
		$compiler_flags = "-g -O0";
		last;
	} elsif($line eq "4" ){
 		$compiler_flags = "--with-production --enable-tracing";
		last;
	} elsif($line eq "3" || $line eq ""){ 
                $compiler_flags = "--with-production";
                last; 
        }  elsif($line eq "5"){

		print "Enter compiler options: ";
		my $input_line = <>;
		chomp($input_line);
		$compiler_flags = $input_line;
		
		last;
	} else {
		print "Invalid option, please try again :P\n"
	}
}




# Determine the target to build.
# We want this simple so we just give 2 options
my $target = "";

print << "EOF";

What do you want to build?
	1) Charm++ [default] (choose this if you are building NAMD)
	2) Charm++ and AMPI
	3) Charm++, AMPI, ParFUM, FEM and other libraries

EOF

while(my $line = <>){
	chomp $line;
	if($line eq "1" || $line eq ""){
		$target = "charm++";
		last;
	} elsif($line eq "2"){
		$target = "AMPI";
		last;
	} elsif($line eq "3"){
		$target = "LIBS";
		last;
	} else {
		print "Invalid option, please try again :P\n"
	}
	
}

# Determine whether to use a -j flag for faster building
my $j = "";
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

    while(my $line = <>) {
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
my $build_line = "./build $target $arch $compilers $options $j $nobs ${compiler_flags}\n";


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
my $p = promptUserYN();
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



