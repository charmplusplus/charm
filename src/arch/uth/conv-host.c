/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>

usage()
{
  fprintf(stderr,"usage: conv-host <pgm> <arguments>\n");
  exit(1);
}

main(argc, argv)
int argc;
char **argv;
{
  if (argc<2) usage();
  execvp(argv[1],argv+1);
  fprintf(stderr,"not found: %s\n",argv[1]);
}
