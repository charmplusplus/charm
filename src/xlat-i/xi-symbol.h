#ifndef _SYMBOL_H
#define _SYMBOL_H

#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "xi-util.h"

class Value : public Printable {
  private:
    char *val;
  public:
    Value(char *s) : val(s) {}
    void print(XStr& str) { str << val; }
};

class ValueList : public Printable {
  private:
    Value *val;
    ValueList *next;
  public:
    ValueList(Value* v, ValueList* n=0) : val(v), next(n) {}
    void print(XStr& str) {
      if(val) {
        str << "["; val->print(str); str << "]";
      }
      if(next)
        next->print(str);
    }
};

class Construct : public Printable {
  protected:
    int external;
  public:
    void setExtern(int e) { external = e; }
    virtual void genDecls(XStr& str) = 0;
    virtual void genDefs(XStr& str) = 0;
    virtual void genReg(XStr& str) = 0;
};

class ConstructList : public Construct {
    Construct *construct;
    ConstructList *next;
  public:
    ConstructList(Construct *c, ConstructList *n=0) : construct(c), next(n) {}
    void setExtern(int e);
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

/* Template Instantiation Parameter */

class TParam : public Printable {
  public:
    virtual void genSpec(XStr& str)=0;
};

/* List of Template Instantiation parameters */

class TParamList : public Printable {
    TParam *tparam;
    TParamList *next;
  public:
    TParamList(TParam *t, TParamList *n=0) : tparam(t), next(n) {}
    void print(XStr& str);
    void genSpec(XStr& str);
};

class Type : public Printable {
  public:
    virtual void print(XStr&) = 0;
    virtual int isVoid(void) = 0;
    virtual char *getBaseName(void) = 0;
    virtual void genProxyName(XStr&) = 0;
};

class TypeList : public Printable {
    Type *type;
    TypeList *next;
  public:
    TypeList(Type *t, TypeList *n=0) : type(t), next(n) {}
    void print(XStr& str);
    void genProxyNames(XStr& str, char*, char*, char*);
};

/* EnType is the type of an entry method parameter, 
   or return type of an entry method
*/

class EnType : virtual public Type {
  public:
    virtual void genMsgProxyName(XStr& str) = 0;
};

class SimpleType : virtual public Type {
};

class BuiltinType : public SimpleType , public EnType {
  private:
    char *name;
  public:
    BuiltinType(char *n) : name(n) {}
    void print(XStr& str) { str << name; }
    int isVoid(void) { return !strcmp(name, "void"); }
    char *getBaseName(void) { return name; }
    void genProxyName(XStr& str) { cerr << "Illegal Base Class ?\n"; abort(); }
    void genMsgProxyName(XStr& str) { 
      cerr << "Illegal Entry Param?\n"; 
      abort(); 
    }
};

class NamedType : public SimpleType {
  private:
    char *name;
    TParamList *tparams;
  public:
    NamedType(char* n, TParamList* t=0) : name(n), tparams(t) {}
    void print(XStr& str) { str << name; }
    int isVoid(void) { return 0; }
    char *getBaseName(void) { return name; }
    int isTemplated(void) { return (tparams!=0); }
    void genProxyName(XStr& str) { genChareProxyName(str); }
    void genChareProxyName(XStr& str) { str << chare_prefix() << name; }
    void genGroupProxyName(XStr& str) { str << group_prefix() << name; }
    void genArrayProxyName(XStr& str) { str << array_prefix() << name; }
    void genMsgProxyName(XStr& str) { str << msg_prefix() << name; }
};

class PtrType : public EnType {
  private:
    Type *type;
    int numstars; // level of indirection
  public:
    PtrType(Type *t) : type(t), numstars(1) {}
    void indirect(void) { numstars++; }
    void print(XStr& str);
    int isVoid(void) { return 0; }
    char *getBaseName(void) { return type->getBaseName(); }
    void genProxyName(XStr& str) { cerr << "Illegal Base Class ?\n"; abort(); }
    void genMsgProxyName(XStr& str) { 
      if(numstars != 1) {
        cerr << "Illegal Entry Param ?\n"; 
        abort(); 
      } else {
        str << msg_prefix();
        type->print(str);
      }
    }
};

class ArrayType : public Type {
  private:
    Type *type;
    Value* dim;
  public:
    ArrayType(Type* t, Value* d) : type(t), dim(d) {}
    void print(XStr& str){type->print(str);str<<"[";dim->print(str);str<<"]";}
    int isVoid(void) { return 0; }
    char *getBaseName(void) { return type->getBaseName(); }
    void genProxyName(XStr& str) { cerr << "Illegal Base Class ?\n"; abort(); }
};

class FuncType : public Type {
  private:
    Type *rtype;
    char *name;
    TypeList *tlist;
  public:
    FuncType(Type* r, char* n, TypeList* t) : rtype(r), name(n), tlist(t) {}
    void print(XStr& str) { 
      rtype->print(str);
      str << "(*" << name << ")(";
      if(tlist)
        tlist->print(str);
    }
    int isVoid(void) { return 0; }
    char *getBaseName(void) { return name; }
    void genProxyName(XStr& str) { cerr << "Illegal Base Class ?\n"; abort(); }
};

/* A type instantiation parameter */

class TParamType : public TParam {
  Type *type;
  public:
    TParamType(Type *t) : type(t) {}
    void print(XStr& str) { type->print(str); }
    void genSpec(XStr& str) { type->print(str); }
};

/* A Value instantiation parameter */

class TParamVal : public TParam {
    char *val;
  public:
    TParamVal(char *v) : val(v) {}
    void print(XStr& str) { str << val; }
    void genSpec(XStr& str) { str << val; }
};

class Chare;

/* Member of a chare or group, i.e. entry, RO or ROM */

class Member : public Printable {
  protected:
    Chare *container;
  public:
    void setChare(Chare *c) { container = c; }
    virtual void genDecls(XStr& str) = 0;
    virtual void genDefs(XStr& str) = 0;
    virtual void genReg(XStr& str) = 0;
};

/* List of members of a chare or group */

class MemberList : public Printable {
    Member *member;
    MemberList *next;
  public:
    MemberList(Member *m, MemberList *n=0) : member(m), next(n) {}
    void print(XStr& str);
    void setChare(Chare *c);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

/* A template construct */

class TVarList;
class TEntity;

class Template : public Construct {
    TVarList *tspec;
    TEntity *entity;
  public:
    Template(TVarList *t, TEntity *e) : tspec(t), entity(e) {}
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void genSpec(XStr& str);
    void genVars(XStr& str);
};


/* An entity that could be templated, i.e. chare, group or a message */

class TEntity : public Printable {
  protected:
    Template *templat;
  public:
    void setTemplate(Template *t) { templat = t; }
    virtual void genDecls(XStr& str) = 0;
    virtual void genDefs(XStr& str) = 0;
    virtual void genReg(XStr& str) = 0;
    void genSpec(XStr& str) { templat->genSpec(str); }
    void genVars(XStr& str) { templat->genVars(str); }
};

#define SCHARE 1
#define SMAINCHARE 2
#define SGROUP 3
#define SARRAY 4
#define SNODEGROUP 5

/* Chare or group is a templated entity */

class Chare : public TEntity, public Construct {
    int chareType; // is chare/mainchare or group
    NamedType *type;
    MemberList *list;
    TypeList *bases;
  public:
    Chare(int c, NamedType *t, TypeList *b=0, MemberList *l=0) : 
      chareType(c), type(t), bases(b), list(l) {setTemplate(0);}
    int  getChareType(void) { return chareType; }
    void genProxyName(XStr& str){type->genProxyName(str);}
    void genProxyBases(XStr& str, char* p, char* s, char* sep) {
      bases->genProxyNames(str, p, s, sep);
    }
    char *getBaseName(void) { return type->getBaseName(); }
    int  isTemplated(void) { return (templat!=0); }
    int  isDerived(void) { return (bases!=0); }
    void print(XStr& str);
    void genChareDecls(XStr& str);
    void genGroupDecls(XStr& str);
    void genArrayDecls(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

#define SPACKED  0x01
#define SVARSIZE 0x02

class Message : public TEntity, public Construct {
    int attrib;
    NamedType *type;
  public:
    Message(NamedType *t, int a) : type(t), attrib(a) { setTemplate(0); }
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    int  isPacked(void) { return attrib&SPACKED; }
    int  isVarsize(void) { return attrib&SVARSIZE; }
};

/* A formal argument of a template */

class TVar : public Printable {
  public:
    virtual void genLong(XStr& str) = 0;
    virtual void genShort(XStr& str) = 0;
};

/* a formal type argument */

class TType : public TVar {
    Type *type;
    Type *init;
  public:
    TType(Type *t, Type *i=0) : type(t), init(i) {}
    void print(XStr& str);
    void genLong(XStr& str);
    void genShort(XStr& str);
};

/* a formal function argument */

class TFunc : public TVar {
    FuncType *type;
    char *init;
  public:
    TFunc(FuncType *t, char *v=0) : type(t), init(v) {}
    void print(XStr& str) { type->print(str); if(init) str << "=" << init; }
    void genLong(XStr& str){ type->print(str); if(init) str << "=" << init; }
    void genShort(XStr& str) {str << type->getBaseName(); }
};

/* A formal variable argument */

class TName : public TVar {
    Type *type;
    char *name;
    char *val;
  public:
    TName(Type *t, char *n, char *v=0) : type(t), name(n), val(v) {}
    void print(XStr& str);
    void genLong(XStr& str);
    void genShort(XStr& str);
};

/* A list of formal arguments to a template */

class TVarList : public Printable {
    TVar *tvar;
    TVarList *next;
  public:
    TVarList(TVar *v, TVarList *n=0) : tvar(v), next(n) {}
    void print(XStr& str);
    void genLong(XStr& str);
    void genShort(XStr& str);
};

/* An entry construct */

#define STHREADED 0x01
#define SSYNC     0x02
#define SLOCKED   0x04

class Entry : public Member {
  private:
    int attribs;
    EnType *retType;
    char *name;
    EnType *param;
    Value *stacksize;
    void genEpIdx(XStr& str);
    void genEpIdxDecl(XStr& str);
    void genEpIdxDef(XStr& str);
    void genChareStaticConstructorDecl(XStr& str);
    void genChareDecl(XStr& str);
    void genGroupStaticConstructorDecl(XStr& str);
    void genArrayStaticConstructorDecl(XStr& str);
    void genGroupDecl(XStr& str);
    void genArrayDecl(XStr& str);
    void genChareStaticConstructorDefs(XStr& str);
    void genChareDefs(XStr& str);
    void genGroupStaticConstructorDefs(XStr& str);
    void genArrayStaticConstructorDefs(XStr& str);
    void genGroupDefs(XStr& str);
    void genArrayDefs(XStr& str);
  public:
    Entry(int a, EnType *r, char *n, EnType *p, Value *sz=0) :
      attribs(a), retType(r), name(n), param(p), stacksize(sz)
    { setChare(0); 
      if(!isThreaded() && stacksize) {
        cerr << "Non-Threaded methods cannot have stacksize spec.!!\n";
        abort();
      }
      if(retType && !isSync() && !retType->isVoid()) {
        cerr << "Async methods cannot have non-void return type!!\n";
        abort();
      }
    }
    int isThreaded(void) { return (attribs & STHREADED); }
    int isSync(void) { return (attribs & SSYNC); }
    int isConstructor(void) { return !strcmp(name, container->getBaseName());}
    int isExclusive(void) { return (attribs & SLOCKED); }
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

class Module : public Construct {
    int _isMain;
    char *name;
    ConstructList *clist;
  public:
    Module(char *n, ConstructList *c) : name(n), clist(c) { _isMain=0; }
    void print(XStr& str);
    void generate(void);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void setMain(void) { _isMain = 1; }
    int isMain(void) { return _isMain; }
};

class ModuleList : public Printable {
    Module *module;
    ModuleList *next;
  public:
    ModuleList(Module *m, ModuleList *n=0) : module(m), next(n) {}
    void print(XStr& str);
    void generate(void);
};

class Readonly : public Construct, public Member {
    int msg; // is it a readonly var(0) or msg(1) ?
    Type *type;
    char *name;
    ValueList *dims;
  public:
    Readonly(Type *t, char *n, ValueList* d, int m=0) : type(t), name(n), msg(m)
    { dims=d; setChare(0); }
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

#endif
