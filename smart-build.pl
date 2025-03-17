#!/usr/bin/env perl


# This is an interactive script that knows
# common ways to build Charm++ and AMPI.
#
# Authors: dooley, becker

use strict;
use warnings;

# Get location of script
use File::Basename;
my $dirname = dirname(__FILE__);

# Create temporary file for compiler tests
use File::Temp qw(tempfile);
my $tempfile = new File::Temp(UNLINK => 1, SUFFIX => '.c');
print $tempfile "\n";

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
print "\nInteractive Charm++/AMPI configuration ...\n";
print "If you are a power user expecting a list of options, please use ./build --help\n";
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
}


my $x86;
my $amd64;
my $ppc;
my $arm7;
my $arm8;
# Determine architecture (x86, ppc, ...)
if($cpu =~ m/i[0-9]86/){
  $x86 = 1;
} elsif($cpu =~ m/x86\_64/){
  $amd64 = 1;
} elsif($cpu =~ m/powerpc/){
  $ppc = 1;
} elsif($cpu =~ m/ppc*/){
  $ppc = 1;
} elsif($cpu =~ m/aarch64*/ || $cpu =~ m/arm64*/){
  $arm8 = 1;
} elsif($cpu =~ m/arm7/ || $cpu =~ m/armv7*/ || $cpu =~ m/armv6*/){
  $arm7 = 1;
}


# default to netlrts
my $converse_network_type = "netlrts";
my $skip_choosing = "false";

print "Are you building to run just on the local machine, and not across multiple nodes? [";
if($arch_os eq "darwin") {
    print "Y/n]: ";
} else {
    print "y/N]: ";
}
{
    my $p = promptUserYN();
    if($p eq "yes" || ($arch_os eq "darwin" && $p eq "default")){
	$converse_network_type = "multicore";
	$skip_choosing = "true";
    }
}


# check for Cray

if($skip_choosing eq "false"){
  my $craycc_found = index(`which CC 2>/dev/null`, "/opt/cray/") != -1;

  my $PE_PRODUCT_LIST = $ENV{'PE_PRODUCT_LIST'};
  if (not defined $PE_PRODUCT_LIST) {
    $PE_PRODUCT_LIST = "";
  }

  my $CRAYPE_NETWORK_TARGET = $ENV{'CRAYPE_NETWORK_TARGET'};
  if (not defined $CRAYPE_NETWORK_TARGET) {
    $CRAYPE_NETWORK_TARGET = "";
  }

  my $CRAY_UGNI_found = index(":$PE_PRODUCT_LIST:", ":CRAY_UGNI:") != -1;

  my $gni_found = $craycc_found || $CRAY_UGNI_found;

  if ($CRAYPE_NETWORK_TARGET eq "ofi") {
    print "\nI found that you have a Cray environment.\nDo you want to build Charm++ targeting Cray Shasta? [Y/n]: ";
    my $p = promptUserYN();
    if($p eq "yes" || $p eq "default") {
                  $arch = "ofi-crayshasta";
                  $skip_choosing = "true";
    }
  } elsif ($gni_found) {
    my $CRAYPE_INTERLAGOS_found = index(":$PE_PRODUCT_LIST:", ":CRAYPE_INTERLAGOS:") != -1;
    if ($CRAYPE_INTERLAGOS_found) {
      print "\nI found that you have a Cray environment with Interlagos processors.\nDo you want to build Charm++ targeting Cray XE? [Y/n]: ";
      my $p = promptUserYN();
      if($p eq "yes" || $p eq "default") {
                    $arch = "gni-crayxe";
                    $skip_choosing = "true";
      }
    } else {
      print "\nI found that you have a Cray environment.\nDo you want to build Charm++ targeting Cray XC? [Y/n]: ";
      my $p = promptUserYN();
      if($p eq "yes" || $p eq "default") {
                    $arch = "gni-crayxc";
                    $skip_choosing = "true";
      }
    }
  }
}


# check for OFI

if($skip_choosing eq "false"){
  my $ofi_found = index(`cc $tempfile -Wl,-lfabric 2>&1`, "-lfabric") == -1;

  if ($ofi_found) {
    print "\nI found that you have libfabric available in your toolchain.\nDo you want to build Charm++ targeting OFI? [Y/n]: ";
    my $p = promptUserYN();
    if($p eq "yes" || $p eq "default") {
      $converse_network_type = "ofi";
      $skip_choosing = "true";
    }
  }
}


# check for PAMI

if($skip_choosing eq "false"){
  my $MPI_ROOT = $ENV{'MPI_ROOT'};
  if (not defined $MPI_ROOT) {
    $MPI_ROOT = "";
  }

  my $pami_found = index(`cc $tempfile -Wl,-L,"$MPI_ROOT/lib/pami_port" -Wl,-L,/usr/lib/powerpc64le-linux-gnu -Wl,-lpami 2>&1`, "-lpami") == -1;

  if ($pami_found) {
    print "\nI found that you have libpami available in your toolchain.\nDo you want to build Charm++ targeting PAMI? [Y/n]: ";
    my $p = promptUserYN();
    if($p eq "yes" || $p eq "default") {
      $converse_network_type = "pamilrts";
      $skip_choosing = "true";
    }
  }
}


# check for UCX

if($skip_choosing eq "false"){
  my $ucx_found = index(`cc $tempfile -Wl,-lucp 2>&1`, "-lucp") == -1;

  if ($ucx_found) {
    print "\nI found that you have UCX libs available in your toolchain.\nDo you want to build Charm++ targeting UCX? [Y/n]: ";
    my $p = promptUserYN();
    if($p eq "yes" || $p eq "default") {
      $converse_network_type = "ucx";
      $skip_choosing = "true";
    }
  }
}


# check for Verbs

if($skip_choosing eq "false"){
  my $verbs_found = index(`cc $tempfile -Wl,-libverbs 2>&1`, "-libverbs") == -1;

  if ($verbs_found) {
    print "\nI found that you have libibverbs available in your toolchain.\nDo you want to build Charm++ targeting Infiniband Verbs? [Y/n]: ";
    my $p = promptUserYN();
    if($p eq "yes" || $p eq "default") {
      $converse_network_type = "verbs";
      $skip_choosing = "true";
    }
  }
}


# check for MPI

if($skip_choosing eq "false"){
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
  if($mpi_found eq "true"){
    print "\nI found that you have an mpicc available in your path.\nDo you want to build Charm++ on this MPI? [y/N]: ";
    my $p = promptUserYN();
    if($p eq "yes"){
    $converse_network_type = "mpi";
    $skip_choosing = "true";
    $options = "$options $mpioption";
    }
  }
}


if($skip_choosing eq "false") {

  print "\nDo you have a special network interconnect? [y/N]: ";
  my $p = promptUserYN();
  if($p eq "yes"){

	print << "EOF";

Choose an interconnect from below: [1-10]
	 1) MPI
	 2) Infiniband (verbs)
	 3) Cray XE, XK
	 4) Cray XC
	 5) Intel Omni-Path (ofi)
	 6) PAMI
	 7) UCX

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
		$converse_network_type = "ofi";
		last;
	  } elsif($line eq "6"){
		$converse_network_type = "pamilrts";
		last;
	  } elsif($line eq "7"){
		$converse_network_type = "ucx";
		last;
	  } else {
		print "Invalid option, please try again :P\n"
	  }
	}
  }
}


# check for CUDA

my $nvcc_found = "false";
my $n = system("which nvcc > /dev/null 2>/dev/null") / 256;
if($n == 0){
  $nvcc_found = "true";
}

if($nvcc_found eq "true"){
  print "\nI found that you have NVCC available in your path.\nDo you want to build Charm++ with GPU Manager support for CUDA? [y/N]: ";
  my $p = promptUserYN();
  if($p eq "yes") {
    $options = "$options cuda";
  }
}


# construct an $arch string if we did not explicitly set one above
if($arch eq ""){
  $arch = "${converse_network_type}-${arch_os}";
	  if($amd64) {
		$arch = $arch . "-x86_64";
	  } elsif($ppc){
		$arch = $arch . "-ppc64le";
	  } elsif($arm8){
	  	$arch = $arch . "-arm8";
	  } elsif($arm7){
	  	$arch = $arch . "-arm7";
	  }
}


#================ Choose SMP/PXSHM =================================

# find what options are available
my $opts = `$dirname/buildold charm++ $arch help 2>&1 | grep "Supported options"`;
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
    print "\nHow do you want to handle SMP/Multicore: [1-$counter]\n";
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
my $cs = `$dirname/buildold charm++ $arch help 2>&1 | grep "Supported compilers"`;
# prune away beginning of the line
$cs =~ m/Supported compilers: (.*)/;
$cs = $1;
# split the line into an array
my @c_list = split(" ", $cs);

# print list of compilers
my $numc = @c_list;

if ($numc > 0) {
    print "\nDo you want to specify a compiler? [y/N]: ";
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
$explanations{"ooc"} = "Enable Out-of-core execution support in Charm++";
$explanations{"tcp"} = "Charm++ over TCP instead of UDP for net versions. TCP is slower";
$explanations{"gfortran"} = "Use the gfortran compiler for Fortran";
$explanations{"flang"} = "Use the flang compiler for Fortran";
$explanations{"ifort"} = "Use Intel's ifort Fortran compiler";
$explanations{"pgf90"} = "Use Portland Group's pgf90 Fortran compiler";
$explanations{"syncft"} = "Use fault tolerance support";
$explanations{"omp"} = "Build Charm++ with integrated OpenMP support";
$explanations{"papi"} = "Enable PAPI performance counters";
$explanations{"nolb"} = "Build without load balancing support";
$explanations{"perftools"} = "Build with support for the Cray perftools";
$explanations{"persistent"} = "Build the persistent communication interface";
$explanations{"simplepmi"} = "Use simple PMI for task launching";
$explanations{"slurmpmi"} = "Use Slurm PMI for task launching";
$explanations{"slurmpmi2"} = "Use Slurm PMI2 for task launching";
$explanations{"ompipmix"} = "Use Open MPI PMIX for task launching";
$explanations{"openpmix"} = "Use OpenPMIx for task launching";
$explanations{"tsan"} = "Compile Charm++ with support for Thread Sanitizer";





  # Produce list of options

  $opts = `$dirname/build charm++ $arch help 2>&1 | grep "Supported options"`;
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

      print "\nDo you want to specify any Charm++ build options, such as Fortran compilers? [y/N]: ";
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
              print "$exp";
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
	3) Charm++, AMPI, ParFUM and other libraries

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
my $build_line = "$dirname/build $target $arch $compilers $options $j $nobs ${compiler_flags}\n";


# Save the build line in the log
open(BUILDLINE, ">>smart-build.log");
print BUILDLINE `date`;
print BUILDLINE "Using the following build command:\n";
print BUILDLINE "$build_line\n";
close(BUILDLINE);


print "We have determined a suitable build line is:\n";
print "\t$build_line\n\n";


# Execute the build line if the appropriate architecture directory exists
print "Do you want to start the build now? [Y/n]: ";
my $p = promptUserYN();
if($p eq "yes" || $p eq "default"){
  if(-e "$dirname/src/arch/$arch"){
	print "Building with: ${build_line}\n";
	# Execute the build line
	system($build_line);
  } else {
	print "We could not figure out how to build charm with those options on this platform, please manually build\n";
	print "Try something similar to: ${build_line}\n";
  }
}



