/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "xi-symbol.h"
#include "xi-grammar.tab.h"

extern FILE *yyin;
extern void yyrestart ( FILE *input_file );
extern int yyparse (void);
extern int yyerror(char *);

extern ModuleList *modlist;

ModuleList *Parse(char *interfacefile)
{
  FILE * fp = fopen (interfacefile, "r") ;
  if (fp) {
    yyin = fp ;
    if(yyparse())
      exit(1);
    fclose(fp) ;
  } else {
    cout << "ERROR : could not open " << interfacefile << endl ;
  }
  return modlist;
}

void abortxi(char *name)
{
  cout << "Usage : " << name << " [-ansi]  module.ci" << endl;
  exit(1) ;
}

main(int argc, char *argv[])
{
  char *fname;
  char *option=0;

  compilemode = original;
  fortranMode = 0;

  switch (argc) {
  case 2:
    fname = argv[1];
    break;

  case 3:
    if (*argv[1]=='-') {
      option = argv[1];
      fname = argv[2];
    } else if (*(argv[2]) == '-') {
      fname = argv[1];
      option = argv[2];
    } else abortxi(argv[0]);

    break;
  default:
    abortxi(argv[0]);
  }
  
  if (option != 0 && strcmp(option,"-ansi")==0)
    compilemode = ansi;

  if (option != 0 && strcmp(option,"-f90")==0)
    fortranMode = 1;

  ModuleList *m = Parse(fname) ;
  m->generate();
  return 0 ;
}
