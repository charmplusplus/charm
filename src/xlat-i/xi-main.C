/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "xi-symbol.h"
#include <string>

using std::cout;
using std::endl;

extern FILE *yyin;
extern void yyrestart ( FILE *input_file );
extern int yyparse (void);
extern int yyerror(char *);

extern xi::ModuleList *modlist;

namespace xi {

#include "xi-grammar.tab.h"

/******************* Macro defines ****************/
class MacroDefinition {
public:
  char *key;
  char *val;
  MacroDefinition(): key(NULL), val(NULL) {}
  MacroDefinition(char *k, char *v): key(k), val(v) {}
  MacroDefinition(char *str) {
    // split by '='
    char *equal = strchr(str, '=');
    if (equal) {
      *equal = 0;
      key = str;
      val = equal+1;
    }
    else {
      key = str;
      val = (char*)"";
    }
  }
  char *match(const char *k) { if (!strcmp(k, key)) return val; return NULL; }
};

static TList<MacroDefinition *> macros;

int macroDefined(const char *str, int istrue)
{
  MacroDefinition *def;
  for (def = macros.begin(); !macros.end(); def=macros.next()) {
    char *val = def->match(str);
    if (val) {
      if (!istrue) return 1;
      else return atoi(val);
    }
  }
  return 0;
}

// input: name
// output: basename (pointer somewhere inside name)
//         scope (null if name is unscoped, newly allocated string otherwise)
void splitScopedName(const char* name, const char** scope, const char** basename) {
    const char* scopeEnd = strrchr(name, ':');
    if (!scopeEnd) {
        *scope = NULL;
        *basename = name;
        return;
    }
    *basename = scopeEnd+1;
    int len = scopeEnd-name+1; /* valid characters to copy */
    char *tmp = new char[len+1];
    strncpy(tmp, name, len);
    tmp[len]=0; /* gotta null-terminate C string */
    *scope = tmp;
}

FILE *openFile(char *interfacefile)
{
  if (interfacefile == NULL) {
    cur_file = "STDIN";
    return stdin;
  }
  else {
    cur_file=interfacefile;
    FILE *fp = fopen (interfacefile, "r") ;
    if (fp == NULL) {
      cout << "ERROR : could not open " << interfacefile << endl;
      exit(1);
    }
    return fp;
  }
  return NULL;
}

/*
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
*/

ModuleList *Parse(FILE *fp)
{
  modlist = NULL;
  yyin = fp ;
  if(yyparse())
      exit(1);
  fclose(fp) ;
  return modlist;
}


void abortxi(char *name)
{
  cout << "Usage : " << name << " [-ansi|-f90|-intrinsic|-M]  module.ci" << endl;
  exit(1) ;
}

}

using namespace xi;

int main(int argc, char *argv[])
{
  char *fname=NULL;
  fortranMode = 0;
  internalMode = 0;
  bool dependsMode = false;

  for (int i=1; i<argc; i++) {
    if (*argv[i]=='-') {
      if (strcmp(argv[i],"-ansi")==0);
      else if (strcmp(argv[i],"-f90")==0)  fortranMode = 1;
      else if (strcmp(argv[i],"-intrinsic")==0)  internalMode = 1;
      else if (strncmp(argv[i],"-D", 2)==0)  macros.append(new MacroDefinition(argv[i]+2));
      else if (strncmp(argv[i], "-M", 2)==0) dependsMode = true;
      else abortxi(argv[0]);
    }
    else
      fname = argv[i];
  }
  //if (fname==NULL) abortxi(argv[0]);

  ModuleList *m = Parse(openFile(fname)) ;
  if (!m) return 0;
  m->preprocess();
  if (dependsMode)
  {
      std::string ciFileBaseName = fname;
      size_t loc = ciFileBaseName.rfind('/');
      if(loc != std::string::npos)
          ciFileBaseName = ciFileBaseName.substr(loc+1);
      m->genDepends(ciFileBaseName);
  }
  else
      m->generate();
  return 0 ;
}
