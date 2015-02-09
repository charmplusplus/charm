#ifndef _TYPE_H
#define _TYPE_H

#include "xi-util.h"

namespace xi {

class TParamList;
class  ParamList;

/*********************** Type System **********************/
class Type : public Printable {
 public:
  virtual void print(XStr&) = 0;
  virtual int isVoid(void) const {return 0;}
  virtual int isBuiltin(void) const { return 0; }
  virtual int isMessage(void) const {return 0;}
  virtual int isTemplated(void) const { return 0; }
  virtual int isPointer(void) const {return 0;}
  virtual int isNamed(void) const { return 0; }
  virtual int isCkArgMsgPtr(void) const {return 0;}
  virtual int isCkArgMsg(void) const {return 0;}
  virtual int isCkMigMsgPtr(void) const {return 0;}
  virtual int isCkMigMsg(void) const {return 0;}
  virtual int isReference(void) const {return 0;}
  virtual int isInt(void) const { return 0; }
  virtual bool isConst(void) const {return false;}
  virtual Type *deref(void) {return this;}
  virtual const char *getBaseName(void) const = 0;
  virtual const char *getScope(void) const = 0;
  virtual int getNumStars(void) const {return 0;}
  virtual void genProxyName(XStr &str,forWhom forElement);
  virtual void genIndexName(XStr &str);
  virtual void genMsgProxyName(XStr& str);
  XStr proxyName(forWhom w)
  	{XStr ret; genProxyName(ret,w); return ret;}
  XStr indexName(void) 
  	{XStr ret; genIndexName(ret); return ret;}
  XStr msgProxyName(void) 
  	{XStr ret; genMsgProxyName(ret); return ret;}
  virtual void printVar(XStr &str, char *var) {print(str); str<<" "; str<<var;}
  int operator==(const Type &tp) const {
    return  (strcmp(getBaseName(), tp.getBaseName())==0);
  }
  virtual ~Type() { }
};

class BuiltinType : public Type {
 private:
  char *name;
 public:
  BuiltinType(const char *n) : name((char *)n) {}
  int isBuiltin(void) const {return 1;}
  void print(XStr& str) { str << name; }
  int isVoid(void) const { return !strcmp(name, "void"); }
  int isInt(void) const { return !strcmp(name, "int"); }
  const char *getBaseName(void) const { return name; }
  const char *getScope(void) const { return NULL; }
};

class NamedType : public Type {
 private:
  const char* name;
  const char* scope;
  TParamList *tparams;

 public:
  NamedType(const char* n, TParamList* t=0, const char* scope_=NULL)
     : name(n), scope(scope_), tparams(t) {}
  int isTemplated(void) const { return (tparams!=0); }
  int isCkArgMsg(void) const {return 0==strcmp(name,"CkArgMsg");}
  int isCkMigMsg(void) const {return 0==strcmp(name,"CkMigrateMessage");}
  void print(XStr& str);
  int isNamed(void) const {return 1;}
  virtual const char *getBaseName(void) const { return name; }
  virtual const char *getScope(void) const { return scope; }
  virtual void genProxyName(XStr& str,forWhom forElement);
  virtual void genIndexName(XStr& str);
  virtual void genMsgProxyName(XStr& str);
};

class PtrType : public Type {
 private:
  Type *type;
  int numstars; // level of indirection

 public:
  PtrType(Type *t) : type(t), numstars(1) {}
  int isPointer(void) const {return 1;}
  int isCkArgMsgPtr(void) const {return numstars==1 && type->isCkArgMsg();}
  int isCkMigMsgPtr(void) const {return numstars==1 && type->isCkMigMsg();}
  int isMessage(void) const {return numstars==1 && !type->isBuiltin();}
  void indirect(void) { numstars++; }
  int getNumStars(void) const {return numstars; }
  void print(XStr& str);
  Type* deref(void) { return type; }
  const char *getBaseName(void) const { return type->getBaseName(); }
  const char *getScope(void) const { return NULL; }
  virtual void genMsgProxyName(XStr& str) { 
    if (numstars != 1) {
      die("too many stars-- entry parameter must have form 'MTYPE *msg'"); 
    } else {
      type->genMsgProxyName(str);
    }
  }
};

class ReferenceType : public Type {
 private:
  Type *referant;

 public:
  ReferenceType(Type *t) : referant(t) {}
  int isReference(void) const {return 1;}
  void print(XStr& str) {str<<referant<<" &";}
  virtual Type *deref(void) {return referant;}
  const char *getBaseName(void) const { return referant->getBaseName(); }
  const char *getScope(void) const { return NULL; }
};

class ConstType : public Type {
 private:
  Type *constType;

 public:
  ConstType(Type *t) : constType(t) {}
  void print(XStr& str) {str << "const " << constType;}
  virtual bool isConst(void) const {return true;}
  virtual Type *deref(void) {return constType;}
  const char *getBaseName(void) const { return constType->getBaseName(); }
  const char *getScope(void) const { return NULL; }
};

class FuncType : public Type {
  private:
    Type *rtype;
    const char *name;
    ParamList *params;

  public:
    FuncType(Type* r, const char* n, ParamList* p)
    	:rtype(r),name(n),params(p) {}
    void print(XStr& str);
    const char *getBaseName(void) const { return name; }
    const char *getScope(void) const { return NULL; }
};

//This is used as a list of base classes
class TypeList : public Printable {
public:
    Type *type;
    TypeList *next;
    TypeList(Type *t, TypeList *n=0) : type(t), next(n) {}
    ~TypeList() { delete type; delete next; }
    int length(void) const;
    Type *getFirst(void) {return type;}
    void print(XStr& str);
    void genProxyNames(XStr& str, const char *prefix, const char *middle, 
                        const char *suffix, const char *sep, forWhom forElement);
};

}   // namespace xi

#endif // ifndef _TYPE_H
