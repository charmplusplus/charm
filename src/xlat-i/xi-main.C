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

main(int argc, char *argv[])
{
  if ( argc != 2 ) {
    cout << "Usage : " << argv[0] << " module.ci" << endl;
    exit(1) ;
  }
  ModuleList *m = Parse(argv[1]) ;
  m->generate();
  return 0 ;
}
