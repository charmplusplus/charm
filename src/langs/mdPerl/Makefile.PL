use ExtUtils::MakeMaker;
# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    'NAME'	=> 'mdPerl',
    'VERSION_FROM' => 'mdPerl.pm', # finds $VERSION
    'LIBS'	=> ['-L/Home/g1/milind/charm/net-hp-cc/lib -lconv-core'],   # e.g., '-lm' 
    'DEFINE'	=> '',     # e.g., '-DHAVE_SOMETHING' 
    'INC'	=> '-I/Home/g1/milind/charm/net-hp-cc/include',     # e.g., '-I/usr/include/other' 
);
