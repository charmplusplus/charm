//  DEFINITIONS OF FUNCTIONS IN symbol.h ---------------------------------------

#include "xi-symbol.h"

extern FILE *yyin;
extern void yyrestart ( FILE *input_file );
extern int yyparse (void);

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

Message::Message(char *n, int p)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	packable = p ;
	next = NULL ;
}

Entry::Entry(char *n, char *m, int t, char *r)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;

	msgtype = thismodule->FindMsg(m) ;
	next = NULL ;

	isthreaded = t;
	returnMsg = thismodule->FindMsg(r) ;
}


Chare::Chare(char *n, int cb)
{
	name = new char [strlen(n)+1] ;
	strcpy(name,n) ;
	entries = NULL ;
	chareboc = cb ;
	next = NULL ;
}
void Chare::AddEntry(char *e, char *m, int t, char *r)
{
	Entry *newe = new Entry(e, m, t, r) ;
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


Module *Parse(char *interfacefile)
{
	char *modulename = new char[strlen(interfacefile)+1] ;
	strcpy(modulename, interfacefile) ;
	modulename[strlen(interfacefile)-3] = '\0' ;

	thismodule = new Module(modulename) ;
	delete modulename;

	FILE * fp = fopen (interfacefile, "r") ;
	if (fp) {
		yyin = fp ;
		yyparse() ;
		fclose(fp) ;
	} else {
		cout << "ERROR : could not open " << interfacefile << endl ;
		delete thismodule;
		thismodule = NULL;
	}

	return thismodule;
}

