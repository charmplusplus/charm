#include "sdag/constructs/Constructs.h"
#include "xi-symbol.h"
#include "xi-util.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <list>

using std::cout;
using std::endl;

extern FILE *yyin;
extern void yyrestart(FILE *input_file);
extern int yyparse(void);
extern void yyerror(const char *);
extern int yylex(void);
extern void scan_string(const char *);

extern xi::AstChildren<xi::Module> *modlist;
extern xi::rwentry rwtable[];

using namespace xi;
#include "xi-grammar.tab.h"

namespace xi {

std::vector<std::string> inputBuffer;

int fortranMode, internalMode;
const char *cur_file;

char *fname, *origFile;

void ReservedWord(int token, int fCol, int lCol) {
  char text[300];
  const char *word = 0;
  for (int i = 0; rwtable[i].tok != 0; ++i) {
    if (rwtable[i].tok == token) {
      word = rwtable[i].res;
      break;
    }
  }
  sprintf(text, "Reserved word '%s' used as an identifier", word);
  xi::pretty_msg("error", text, fCol, lCol);
  yyerror(text);
}

/******************* Macro defines ****************/
class MacroDefinition {
 private:
  char *key;
  char *val;

 public:
  MacroDefinition(): key(NULL), val(NULL) {}
  MacroDefinition(char *k, char *v): key(k), val(v) {}
  explicit MacroDefinition(char *str) {
    // split by '='
    char *equal = strchr(str, '=');
    if (equal) {
      *equal = 0;
      key = str;
      val = equal+1;
    } else {
      key = str;
      val = const_cast<char*>("");
    }
  }
  char *match(const char *k) {
    if (!strcmp(k, key)) {
        return val;
    } else {
        return NULL;
    }
  }
};

static std::list<MacroDefinition *> macros;

int macroDefined(const char *str, int istrue) {
  std::list<MacroDefinition *>::iterator def;
  for (def = macros.begin(); def != macros.end(); ++def) {
    char *val = (*def)->match(str);
    if (val) {
      if (!istrue) {
        return 1;
      } else {
        return atoi(val);
      }
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
  tmp[len] = 0; /* gotta null-terminate C string */
  *scope = tmp;
}

/*
FILE *openFile(char *interfacefile) {
  if (interfacefile == NULL) {
    cur_file = "STDIN";
    return stdin;
  } else {
    cur_file = interfacefile;
    FILE *fp = fopen(interfacefile, "r");
    if (fp == NULL) {
      cout << "ERROR : could not open " << interfacefile << endl;
      exit(1);
    }
    return fp;
  }
  return NULL;
}

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

std::string readFile(const char *interfaceFile) {
  // istream::operator== was introduced in C++11.
  // It seems the usual workaround to multiplex cin/ifstream is to use pointers.
  std::istream *in;
  std::string buffer;
  if (interfaceFile) {
    cur_file = interfaceFile;
    in = new std::ifstream(interfaceFile);
  } else {
    cur_file = (origFile != NULL) ? origFile : "STDIN";
    in = &std::cin;
  }

  std::string line;
  while (std::getline(*in, line)) {
    buffer += line + "\n";
    inputBuffer.push_back(line);
  }

  if (interfaceFile)
    delete in;

  return buffer;
}

AstChildren<Module> *Parse(std::string &str) {
  modlist = NULL;
  scan_string(str.c_str());
  if (yyparse())
    exit(1);
  if (num_errors > 0)
    exit(1);
  return modlist;
}

int count_tokens(std::string &str) {
  scan_string(str.c_str());
  int count = 0;
  while (yylex()) count++;
  return count;
}

void abortxi(char *name) {
  cout << "Usage : " << name << " [-ansi|-f90|-intrinsic|-M]  module.ci" << endl;
  exit(1);
}

}   // namespace xi

using namespace xi;

int processAst(AstChildren<Module> *m, const bool chareNames,
               const bool dependsMode, const int fortranMode_,
               const int internalMode_, char* fname_, char* origFile_) {
  // set globals based on input params
  fortranMode = fortranMode_;
  internalMode = internalMode_;
  origFile = origFile_;
  fname = fname_;

  if (!m) return 0;
  m->preprocess();
  m->check();
  if (num_errors != 0)
    exit(1);

  if (chareNames) {
    m->printChareNames();
    return 0;
  }

  if (dependsMode) {
    std::string ciFileBaseName;
    if (fname != NULL) {
      ciFileBaseName = fname;
    } else if (origFile != NULL) {
      ciFileBaseName = origFile;
    } else {
      abortxi(fname);
    }
    size_t loc = ciFileBaseName.rfind('/');
    if (loc != std::string::npos)
        ciFileBaseName = ciFileBaseName.substr(loc+1);
    m->recurse(ciFileBaseName.c_str(), &Module::genDepend);
  } else {
    m->recursev(&Module::generate);
  }

  return 0;
}

int main(int argc, char *argv[])
{
  char* origFile = NULL;
  char* fname = NULL;
  int fortranMode = 0;
  int internalMode = 0;
  bool dependsMode = false;
  bool countTokens = false;
  bool chareNames = false;

  for (int i = 1; i < argc; i++) {
    if (*argv[i] == '-') {
      if (strcmp(argv[i], "-ansi") == 0) {}
      else if (strcmp(argv[i], "-f90") == 0) fortranMode = 1;
      else if (strcmp(argv[i], "-intrinsic") == 0) internalMode = 1;
      else if (strncmp(argv[i], "-D", 2) == 0) macros.push_back(new MacroDefinition(argv[i]+2));
      else if (strncmp(argv[i],  "-M", 2) == 0) dependsMode = true;
      else if (strcmp(argv[i], "-count-tokens") == 0) countTokens = true;
      else if (strcmp(argv[i], "-chare-names") == 0) chareNames = true;
      else if (strcmp(argv[i], "-orig-file") == 0) origFile = argv[++i];
      else abortxi(argv[0]);
    } else {
      fname = argv[i];
    }
  }
  // if (fname==NULL) abortxi(argv[0]);
  std::string buffer = readFile(fname);
  sanitizeComments(buffer);
  sanitizeStrings(buffer);

  if (countTokens) {
    cout << count_tokens(buffer) << endl;
    return 0;
  }

  AstChildren<Module> *m = Parse(buffer);
  return processAst(m, chareNames, dependsMode, fortranMode, internalMode, fname, origFile);
}
