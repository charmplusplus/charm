//  DEFINITIONS OF FUNCTIONS IN symbol.h ---------------------------------------

#include "xi-symbol.h"

extern FILE *yyin;
extern void yyrestart ( FILE *input_file );
extern int yyparse (void);
extern int yyerror(char *);

// Global Variable - used in parse.y
Module *thismodule;  // current module

//------------------------------------------------------------------------------

Table::Table(char *n)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;
  next = 0 ;
}

ReadOnly::ReadOnly(char *n, char *t, int i)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;
  type = new char [strlen(t)+1] ;
  strcpy(type,t) ;
  ismsg = i ;
  next = 0 ;
}

Message::Message(char *n, int p, int a, int e)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;
  packable = p ;
  allocked = a;
  isextern = e ;
  next = 0 ;
}

Entry::Entry(char *n, char *m, int t, char *r, int s)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;

  msgtype = thismodule->FindMsg(m) ;
  if (m != 0 && msgtype == 0) {  // error
    char *mesg = (char *) malloc(strlen(m) + 100);
    strcpy(mesg, m);
    strcat(mesg, " not declared");
    yyerror(mesg);
    exit(1);
  }

  isthreaded = t;

  returnMsg = thismodule->FindMsg(r) ;
  if (r != 0 && returnMsg == 0) {  // error
    char *mesg = (char *) malloc(strlen(m) + 100);
    strcpy(mesg, m);
    strcat(mesg, " not declared");
    yyerror(mesg);
    exit(1);
  }

  stackSize = s;

  next = 0 ;
}


Chare::Chare(char *n, int cb, int e)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;
  entries = 0 ;
  chareboc = cb ;
  isextern = e ;
  next = 0 ;
  numbases = 0;
}

void Chare::AddEntry(char *e, char *m, int t, char *r, int s)
{
  Entry *newe = new Entry(e, m, t, r, s) ;
  newe->next = entries ;
  entries = newe ;
}

void Chare::AddBase(char *bname)
{
  bases[numbases] = new char[strlen(bname)+1];
  strcpy(bases[numbases], bname);
  numbases++;
}

Module::Module(char *n)
{
  name = new char [strlen(n)+1] ;
  strcpy(name,n) ;

  chares = 0 ;
  messages = 0 ;
  readonlys = 0 ;
  tables = 0 ;
}
void Module::AddChare(Chare *c)
{
  curChare = c;
  c->next = 0 ;
  if(chares == 0) {
    chares = c ;
  } else {
    Chare *tmpc = chares;
    while (tmpc->next != 0)
      tmpc = tmpc->next;
    tmpc->next = c;
  }
}
void Module::AddMessage(Message *m)
{
  m->next = 0 ;
  if(messages == 0) {
    messages = m ;
  } else {
    Message *tmpm = messages;
    while (tmpm->next != 0)
      tmpm = tmpm->next;
    tmpm->next = m;
  }
}
void Module::AddReadOnly(ReadOnly *r)
{
  r->next = readonlys ;
  readonlys = r ;
}
void Module::AddTable(Table *t)
{
  t->next = tables ;
  tables = t ;
}
Message *Module::FindMsg(char *msg)
{
  if (msg != 0)
  for ( Message *m=messages; m!=0; m=m->next ) 
    if ( strcmp(m->name,msg) == 0 )
      return m ; 
  return 0 ;
}

char *getmodulename(char *pathname)
{
  char c = '/';
  int i = 0;

  for(i=strlen(pathname)-1; i>=0; i--)
    if (pathname[i] == c)
      break;

  cout << "MODULE " << &(pathname[i+1]) << endl;
  return &(pathname[i+1]);
}


Module *Parse(char *interfacefile)
{
  char *modulename = new char[strlen(interfacefile)+1] ;
  strcpy(modulename, interfacefile) ;
  modulename[strlen(interfacefile)-3] = '\0' ;

  thismodule = new Module(getmodulename(modulename)) ;
  delete modulename;

  FILE * fp = fopen (interfacefile, "r") ;
  if (fp) {
    yyin = fp ;
    if(yyparse())
      exit(1);
    fclose(fp) ;
  } else {
    cout << "ERROR : could not open " << interfacefile << endl ;
    delete thismodule;
    thismodule = 0;
  }

  return thismodule;
}

