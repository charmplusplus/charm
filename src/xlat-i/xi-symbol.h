#ifndef SYMB_H
#define SYMB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>

class Table { 
  public: 
    char *name;
    Table *next;
    int isextern;
    Table(char *n, int e=0);
    int isExtern(void) { return isextern; }
};

class ReadOnly { 
  public: 
    char *name;
    char *type;
    int ismsg;      // 1 is this is a readonly-msg, 0 if a readonly-var
    int isextern;
    int isExtern(void) { return isextern; }
    ReadOnly *next;
    ReadOnly(char *n, char *t, int i, int e=0);
};

class Message {
  public: 
    char *name;
    int packable;   // 1 if this msg type has pack/unpack functions
    int allocked;   // 1 if this msg type has user-defined alloc function
    Message *next;
    int isextern;
    Message(char *n, int p, int a, int e);
    int isExtern() { return isextern ; }
};

class Entry {
  public: 
    char *name;
    int isthreaded;
    Message *returnMsg;
    Message *msgtype;
    int stackSize;
    Entry *next;
    Entry(char *n, char *m, int t=0, char *r=0, int s=0) ;
    int isThreaded() { return isthreaded ; }
    int isMessage() { return msgtype != 0; }
    int isReturnMsg() { return returnMsg != 0; }
    int get_stackSize() { return stackSize; }
};

class Chare {
  public: 
    char *name;
    Entry *entries;
    int chareboc ;
    int numbases;
    char *bases[10]; // right now maxbases is 10, to be replaced by a list
    Chare *next;
    int isextern;
    Chare(char *n, int cb, int e) ;
    void AddEntry(char *e, char *m, int t=0, char *r=0, int s=0) ;
    void AddBase(char *bname);
    int isExtern() { return isextern ; }
};

class Module {
  friend Module *Parse(char *interfacefile);
  public: 
    char *name;
    Chare *chares;
    Chare *curChare;
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
};
#endif

