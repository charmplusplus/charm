//  DEFINITIONS OF FUNCTIONS IN symbol.h ---------------------------------------

#include "xi-symbol.h"

extern FILE *yyin;
extern void yyrestart ( FILE *input_file );
extern int yyparse (void);
extern int yyerror(char *);

// Global Variable - used in parse.y
Module *thismodule;	// current module

//------------------------------------------------------------------------------

Table::Table(char *n)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	next = NULL ;
}

ReadOnly::ReadOnly(char *n, char *t, int i)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	type = new char [strlen(t)+1] ;
	strcpy(type,t) ;
	ismsg = i ;
	next = NULL ;
}

Message::Message(char *n, int p, int a, int e)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	packable = p ;
        allocked = a;
	isextern = e ;
	next = NULL ;
}

Entry::Entry(char *n, char *m, int t, char *r, int s)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;

	msgtype = thismodule->FindMsg(m) ;
	if (m != NULL && msgtype == NULL) {	// error
		char *mesg = (char *) malloc(strlen(m) + 100);
		strcpy(mesg, m);
		strcat(mesg, " not declared");
		yyerror(mesg);
		exit(1);
	}

	isthreaded = t;

	returnMsg = thismodule->FindMsg(r) ;
	if (r != NULL && returnMsg == NULL) {	// error
		char *mesg = (char *) malloc(strlen(m) + 100);
		strcpy(mesg, m);
		strcat(mesg, " not declared");
		yyerror(mesg);
		exit(1);
	}

	stackSize = s;

	next = NULL ;
}


Chare::Chare(char *n, int cb, int e)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	entries = NULL ;
	chareboc = cb ;
	isextern = e ;
	next = NULL ;
}
void Chare::AddEntry(char *e, char *m, int t, char *r, int s)
{
	Entry *newe = new Entry(e, m, t, r, s) ;
	newe->next = entries ;
	entries = newe ;
}

Module::Module(char *n)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;

	chares = NULL ;
	messages = NULL ;
	readonlys = NULL ;
	tables = NULL ;
}
void Module::AddChare(Chare *c)
{
	c->next = chares ;
	chares = c ;
}
void Module::AddMessage(Message *m)
{
	m->next = messages ;
	messages = m ;
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
	if (msg != NULL)
	for ( Message *m=messages; m!=NULL; m=m->next ) 
		if ( strcmp(m->name,msg) == 0 )
			return m ; 
	return NULL ;
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
		thismodule = NULL;
	}

	return thismodule;
}

