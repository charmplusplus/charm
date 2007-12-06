#!/usr/bin/perl

$| = 1;

print "Begin interactive charm configuration\n\n";


$os = `uname -s`;
$cpu = `uname -m`;

$nobs = "";

#$cpu = "x86_64\n";

chomp($os);
chomp ($cpu);

$network_option_string = "";
$compiler = "";

# Determine kernel
# linux, darwin, ...
if($os eq "Linux"){
	print "Detected a linux kernel\n";
	$arch_os = "linux";
} elsif($os eq "Darwin"){
	print "Detected a darwin kernel\n";
	$arch_os = "darwin";
} elsif($os =~ m/BSD/ ){
	print "Detected a BSD kernel\n";
	$arch_os = "linux";
}


# Determine architecture
# x86, ppc, ia64, amd64
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
}




# Determine converse architecture
# net, mpi, ...

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

	
$converse_network_type = "net";
	
if($special_network eq "true"){
	#select type of interconnect here
	print << "EOF";
	
Choose an interconnect from below: [1-10]
	1) Infiniband (using OSU MPI)
	2) Infiniband (native layer alpha version)
	3) Myrinet GM
	4) Myrinet MX
	5) Amasso
	6) Cray XT3, XT4 (not yet tested on CNL)
	7) Bluegene/L Native
	8) Bluegene/L MPI
	9) Other Vendor MPI
	10) VMI

Note: Some other less common options can be found by calling "./build --help"

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
			$converse_network_type = "ammasso";
			last;
		} elsif($line eq "6"){
			$converse_network_type = "mpi-crayxt3";
			last;
		} elsif($line eq "7"){
			$arch = "bluegenel";
			$compiler = "xlc";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "8"){
			$converse_network_type = "mpi-bluegenel";
			$compiler = "xlc";
			$nobs = "--no-build-shared";
			last;
		} elsif($line eq "9"){
			$converse_network_type = "mpi";
			last;
		} elsif($line eq "10"){
			$converse_network_type = "vmi";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
		
	}	
	
}


$target = "LIBS";
$arch = "";

print "arch_os: $arch_os\n";
print "converse_network_type: $converse_network_type\n";

if($converse_network_type eq "ammasso" || 
   $converse_network_type eq "bluegenel" ||
   $converse_network_type eq "mpi-bluegenel"||
   $converse_network_type eq "mpi-crayxt3" ) 
   {    
      $arch = $converse_network_type;
   }
else 
   {
	  $arch = "${converse_network_type}-${arch_os}";
          print "arch: $arch\n";
	  if($amd64) {
                $arch = $arch . "-amd64";
	  } elsif($ia64){
	  	$arch = $arch . "-ia64";
	  } elsif($ppc){
	  	$arch = $arch . "-ppc";
	  }
	  
   }
print "arch: $arch\n";



#Cleanup the architectures to match the horrible real world inconsistent src/archs

if($arch eq "net-darwin"){
	$arch = "net-darwin-x86";
} elsif($arch eq "net-darwin-ppc"){
	$arch = "net-ppc-darwin";
} elsif($arch eq "mpi-darwin-ppc"){
	$arch = "mpi-ppc-darwin";
} 


if($compiler ne ""){
	print "We determined that you should use the compiler $compiler\n Do you want to use a different compiler?[Y/N]";
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
	
Choose a compiler from below: [1-14]
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

EOF

	while($line = <>){
		chomp $line;
		if($line eq "1"){
			$compiler = "cc";
			last;
		} elsif($line eq "2"){
			$compiler = "cc64";
			last;
		} elsif($line eq "3"){
			$compiler = "cxx";
			last;
		} elsif($line eq "4"){
			$compiler = "kcc";
			last;
		} elsif($line eq "5"){
			$compiler = "pgcc";
			last;
		} elsif($line eq "6"){
			$compiler = "acc";
			last;
		} elsif($line eq "7"){
			$compiler = "icc";
			last;
		} elsif($line eq "8"){
			$compiler = "ecc";
			last;
		} elsif($line eq "9"){
			$compiler = "gcc3";
			last;
		} elsif($line eq "10"){
			$compiler = "gcc4";
			last;
		} elsif($line eq "11"){
			$compiler = "mpcc";
			last;
		} elsif($line eq "12"){
			$compiler = "pathscale";
			last;
		} elsif($line eq "13"){
			$compiler = "xlc";
			last;
		} elsif($line eq "14"){
			$compiler = "xlc64";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
	}
}

# SMP / Multicore

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




print << "EOF";

What do you want to build?
	1) Charm++ [default] (choose this if you are building NAMD)
	2) Charm++, AMPI, ParFUM, FEM and other libraries

EOF

while($line = <>){
	chomp $line;
	if($line eq "1" || $line eq ""){
		$target = "charm++";
		$j = "-j4";
		last;
	} elsif($line eq "2"){
		$target = "LIBS";
		last;
	} else {
		print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
	}
	
}



if($j ne ""){
	print "Do you want to do a parallel build (-j4)?[Y/n]";
	while($line = <>){
		chomp $line;
		if(lc($line) eq "y" || lc($line) eq "yes" || $line eq ""){
			last;
		} elsif(lc($line) eq "n" || lc($line) eq "no" ){
			$j = "";
			last;
		} else {
			print "Invalid option, DOES NOT COMPUTE, please try again :P\n"
		}
	}
}



$build_line = "./build $target $arch ${network_option_string} $compiler $smp $j ${compiler_flags}\n";

open(BUILDLINE, ">smart-build.log");
print BUILDLINE "Using the following build command:\n$build_line\n";
close(BUILDLINE);

if(-e "src/arch/$arch"){
	print "Building with: ${build_line}\n";	
	system($build_line);
} else {
	print "We could not figure out how to build charm with those options on this platform, please manually build\n";
	print "Try something similar to: ${build_line}\n";
}


