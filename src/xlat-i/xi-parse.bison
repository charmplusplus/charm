
%{

#include "xi-symbol.h"

extern int lineno;
extern int yylex (void) ;
extern Module *thismodule ;

// Local data and functions
int yyerror(char *);

%}

/* Conventions:
	Terminal symbols (tokens) are written in CAPITALS.
*/

%union {
	char *strval;
	int intval;
}

%token BOC
%token CHARE
%token ENTRY
%token MESSAGE
%token PACKMESSAGE
%token READONLY
%token STACKSIZE
%token TABLE
%token THREADED
%token EXTERN
%token <strval>	IDENTIFIER
%token <intval> NUMBER

%type <strval> Id ChareName EntryName MessageName ReadOnlyName TableName
%type <strval> SimpleType PtrType OptionalMessagePtr
%type <intval> OptionalThreaded OptionalExtern OptionalStackSize

%%

File	:	ItemElist
	;

ItemElist:	/* empty */
	|	ItemList
	;

ItemList:	Item
	|	ItemList Item
	;

Item	:	Boc | Chare | Message | ReadOnly | Table |
PackMessage
	;

OptionalExtern
	:	/* empty */
		{ $$ = FALSE; }
	|	EXTERN
		{ $$ = TRUE; }
	;

Boc	:	OptionalExtern BOC ChareName '{'
		{
			Chare *c = new Chare($3, BOC, $1) ;
			delete $3;
			thismodule->AddChare(c) ;
		}
		EntryList '}' ';'
	;

Chare	:	OptionalExtern CHARE ChareName '{'
		{
			Chare *c = new Chare($3, CHARE, $1) ;
			delete $3;
			thismodule->AddChare(c) ;
		}
		EntryList '}' ';'
	;

ChareName:	Id
	;

Id	:	IDENTIFIER
	;

EntryList:	Entry
	|	EntryList Entry
	;

OptionalMessagePtr
	:	/* empty */
		{
			$$ = NULL;
		}
	|	MessageName '*'
	;

OptionalThreaded
	:	/* empty */
		{ $$ = FALSE; }
	|	THREADED
		{ $$ = TRUE; }
	;

OptionalStackSize
	:	/* empty */
		{ $$ = 0; }
	|	STACKSIZE NUMBER
		{ $$ = $2; }
	;

Entry	:	OptionalThreaded OptionalMessagePtr ENTRY EntryName '(' OptionalMessagePtr ')' OptionalStackSize ';'
		{
			thismodule->chares->AddEntry($4, $6, $1, $2, $8) ;
			delete $4; delete $6;
		}
	;

EntryName:	Id
	;

Message	:	OptionalExtern MESSAGE MessageName ';'
		{
			Message *m = new Message($3, 0, $1) ;
			delete $3;
			thismodule->AddMessage(m) ;
		}
	;

PackMessage	:	OptionalExtern PACKMESSAGE MessageName ';'
		{
			Message *m = new Message($3, 1, $1) ;
			delete $3;
			thismodule->AddMessage(m) ;
		}
	;

MessageName:	Id
	;

ReadOnly:	READONLY SimpleType ReadOnlyName ';'
		{
			ReadOnly *r = new ReadOnly($3, $2, 0) ;
			delete $2;
			thismodule->AddReadOnly(r) ;
		}
	|	READONLY PtrType ReadOnlyName ';'
		{
			ReadOnly *r = new ReadOnly($3, $2, 1) ;
			delete $2;
			thismodule->AddReadOnly(r) ;
		}
	;

SimpleType:	Id
	;

PtrType	:	Id '*'
		{
			$$ = strcat(strcpy(new char[2+strlen($1)], $1), "*");
			delete $1;
		}
	;

ReadOnlyName:	Id
	;

Table	:	TABLE TableName ';'
		{
			Table *t = new Table($2) ;
			delete $2;
			thismodule->AddTable(t) ;
		}
	;

TableName:	Id
	;

%%

int yyerror(char *mesg)
{
	cout << "Syntax error at line " << lineno << ": " << mesg << endl;
	return 0;
}

