
#ifndef SYMB_H
#define SYMB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>

/* A file contains one module.
   A module contains a list of chares (normal or BOC), messages, 
   readonly variables and tables.
   A chare has a list of entry methods which have an associated msg type.
   A readonly variable has a type and a name.
   A table has only a name.
*/

#define FALSE 0
#define TRUE (!FALSE)


class Table { 
public: char *name;
        Table *next;

	Table(char *n);
} ;

class ReadOnly { 
public: char *name;
        char *type;
        int ismsg;      // 1 is this is a readonly-msg, 0 if a readonly-var
        ReadOnly *next;

	ReadOnly(char *n, char *t, int i);
} ;

class Message {
public: char *name;
        int packable;   // 1 is this msg type has pack/unpack functions
        Message *next;
	int isextern;

	Message(char *n, int p, int e);
	int isExtern() { return isextern ; }
} ;

class Entry {
public: char *name;
	int isthreaded;
	Message *returnMsg;
        Message *msgtype;
	int stackSize;
        Entry *next;

	Entry(char *n, char *m, int t = FALSE, char *r = NULL, int s = 0) ;
	int isThreaded() { return isthreaded ; }
	int isMessage() { return msgtype != NULL; }
	int isReturnMsg() { return returnMsg != NULL; }
	int get_stackSize() { return stackSize; }
} ;

class Chare {
public: char *name;
        Entry *entries;
	int chareboc ;
        Chare *next;
	int isextern;

	Chare(char *n, int cb, int e) ;
	void AddEntry(char *e, char *m, int t = FALSE, char *r = NULL, int s = 0) ;
	int isExtern() { return isextern ; }
} ;

class Module {
	friend Module *Parse(char *interfacefile);

public: char *name;
        Chare *chares;
        Message *messages;
        ReadOnly *readonlys;
        Table *tables;
	Module *next;

	Module(char *n) ;
	void AddChare(Chare *c) ;
	void AddMessage(Message *m) ;
	void AddReadOnly(ReadOnly *r) ;
	void AddTable(Table *t) ;
	Message *FindMsg(char *msg) ;
} ;

#endif

