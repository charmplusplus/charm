//////////////////////////////////////////////
//
//  rings.h
//
//  Declaration of chares and messages of rings
//
//  Author: Michael Lang
//  Date: 6/15/99
//
///////////////////////////////////////////////

#include "rings.decl.h"

class Token : public CMessage_Token {
public:
  int value;
  int hopSize;
  int loops;
};

class NotifyDone : public CMessage_NotifyDone {
public:
  int value;
};

class main : public Chare {
private:
  int count;
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *m) {}
  void ringDone(NotifyDone *m);
};

class ring : public Group {
private:
  int nextHop;
public:
  ring(Token *t);
  ring(CkMigrateMessage *m) {}
  void passToken(Token *t);
};

  
  
