package mdPerl;

use strict;
use vars qw($VERSION @ISA @EXPORT);

require Exporter;
require DynaLoader;

@ISA = qw(Exporter DynaLoader);
# Items to export into callers namespace by default. Note: do not export
# names by default without a very good reason. Use EXPORT_OK instead.
# Do not simply export all your public functions/methods/constants.
@EXPORT = qw(
	mdInit  mdExit  mdPrintf  mdError  mdMyPe  mdNumPes
	mdScheduler mdExitScheduler mdCall mdTimer
);
$VERSION = '0.01';

bootstrap mdPerl $VERSION;

# Preloaded methods go here.

# Autoload methods go after =cut, and are processed by the autosplit program.

1;
__END__
# Below is the stub of documentation for your module. You better edit it!

=head1 NAME

mdPerl - Perl extension for Message-Driven Parallel Programming

=head1 SYNOPSIS

  use mdPerl;
  ConverseInit();
  CmiPrintf(string);
  ConverseExit();

=head1 DESCRIPTION

Add Some description here.

=head1 AUTHOR

M. A. Bhandarkar, milind@cs.uiuc.edu

=head1 SEE ALSO

perl(1).

=cut
