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
  cur_file=interfacefile;
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
  cout << "Usage : " << name << " [-ansi|-f90|-intrinsic]  module.ci" << endl;
  exit(1) ;
}

main(int argc, char *argv[])
{
  char *fname=NULL;
  fortranMode = 0;
  internalMode = 0;

  for (int i=1; i<argc; i++) {
    if (*argv[i]=='-') {
      if (strcmp(argv[i],"-f90")==0)  fortranMode = 1;
      if (strcmp(argv[i],"-intrinsic")==0)  internalMode = 1;
      else abortxi(argv[0]);
    }
    else
      fname = argv[i];
  }
  if (fname==NULL) abortxi(argv[0]);

  ModuleList *m = Parse(fname) ;
  m->generate();
  return 0 ;
}
