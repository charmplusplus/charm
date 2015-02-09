#ifndef _MESSAGE_H
#define _MESSAGE_H

#include "xi-Template.h"

namespace xi {

class MsgVar {
 public:
  Type *type;
  const char *name;
  int cond;
  int array;
  MsgVar(Type *t, const char *n, int c, int a);
  Type *getType();
  const char *getName();
  int isConditional();
  int isArray();
  void print(XStr &str);
};

class MsgVarList : public Printable {
 public:
  MsgVar *msg_var;
  MsgVarList *next;
  MsgVarList(MsgVar *mv, MsgVarList *n=0);
  void print(XStr &str);
  int len(void);
};

class Message : public TEntity {
  NamedType *type;
  MsgVarList *mvlist;
  void printVars(XStr& str);

 public:
  Message(int l, NamedType *t, MsgVarList *mv=0);
  void print(XStr& str);
  void genDecls(XStr& str);
  void genDefs(XStr& str);
  void genReg(XStr& str);

  virtual const char *proxyPrefix(void);
  void genAllocDecl(XStr& str);
  int numArrays(void);
  int numConditional(void);
  int numVars(void);
};

}   // namespace xi

#endif  // ifndef _MESSAGE_H
