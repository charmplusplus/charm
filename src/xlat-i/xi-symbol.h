#ifndef _SYMBOL_H
#define _SYMBOL_H

#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "xi-util.h"

typedef enum { original, ansi } CompileMode;
extern CompileMode compilemode;

class Value : public Printable {
  private:
    int factor;
    char *val;
  public:
    Value(char *s);
    void print(XStr& str) { str << val; }
    int getIntVal(void);
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
    virtual const char *getBaseName(void) = 0;
    virtual void genProxyName(XStr &str);
};

class TypeList : public Printable {
    Type *type;
    TypeList *next;
  public:
    TypeList(Type *t, TypeList *n=0) : type(t), next(n) {}
    void print(XStr& str);
    void genProxyNames(XStr& str, const char*, const char*, const char*);
    void genProxyNames2(XStr& str, const char*, const char*, 
                        const char*, const char*);
};

/* EnType is the type of an entry method parameter, 
   or return type of an entry method
*/

class EnType : virtual public Type {
  public:
    virtual void print(XStr&) = 0;
    virtual void genMsgProxyName(XStr& str) { 
      cerr << "Illegal genMsgProxy call?\n"; 
      abort(); 
    }
};

class SimpleType : virtual public Type {
};

class BuiltinType : public SimpleType , public EnType {
  private:
    const char *name;
  public:
    BuiltinType(const char *n) : name(n) {}
    void print(XStr& str) { str << name; }
    int isVoid(void) { return !strcmp(name, "void"); }
    const char *getBaseName(void) { return name; }
};

class Chare;//Forward declaration
static char *msg_prefix(void) {return "CMessage_";}
static char *chare_prefix(void) {return "CProxy_";}

class NamedType : public SimpleType {
  private:
    const char *name;
    TParamList *tparams;
  public:
    NamedType(const char* n, TParamList* t=0)
       : name(n), tparams(t) {}
    void print(XStr& str);
    int isVoid(void) { return 0; }
    const char *getBaseName(void) { return name; }
    int isTemplated(void) { return (tparams!=0); }
    virtual void genProxyName(XStr& str) { str << chare_prefix(); print(str);}
    virtual void genMsgProxyName(XStr& str) { str << msg_prefix(); print(str);}
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
    const char *getBaseName(void) { return type->getBaseName(); }
    virtual void genMsgProxyName(XStr& str) { 
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
    const char *getBaseName(void) { return type->getBaseName(); }
};

class FuncType : public Type {
  private:
    Type *rtype;
    const char *name;
    TypeList *tlist;
  public:
    FuncType(Type* r, const char* n, TypeList* t) :rtype(r),name(n),tlist(t) {}
    void print(XStr& str) { 
      rtype->print(str);
      str << "(*" << name << ")(";
      if(tlist)
        tlist->print(str);
    }
    int isVoid(void) { return 0; }
    const char *getBaseName(void) { return name; }
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
    virtual int isPure(void) { return 0; }
    virtual void genDecls(XStr& str) = 0;
    virtual void genDefs(XStr& str) = 0;
    virtual void genReg(XStr& str) = 0;
    XStr makeDecl(const char *returnType);
};

/* List of members of a chare or group */

class MemberList : public Printable {
    Member *member;
    MemberList *next;
  public:
    MemberList(Member *m, MemberList *n=0) : member(m), next(n) {}
    void print(XStr& str);
    void setChare(Chare *c);
    int isPure(void);
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
    virtual void genTSpec(XStr& str) { if (templat) templat->genSpec(str); }
    virtual void genTVars(XStr& str) { if (templat) templat->genVars(str); }
};

/* Chare or group is a templated entity */

class Chare : public TEntity, public Construct {
  protected:
    NamedType *type;
    MemberList *list;
    TypeList *bases;
    int abstract;

    void genRegisterMethodDecl(XStr& str);
    void genRegisterMethodDef(XStr& str);
  public:
    Chare(NamedType *t, TypeList *b=0, MemberList *l=0) : 
      type(t), list(l), bases(b) 
    {
      setTemplate(0); 
      abstract=0;
    }
    void genProxyBases(XStr& str,const char* p,const char* s,const char* sep) {
      bases->genProxyNames(str, p, s, sep);
    }
    XStr proxyName(int withTemplates=1) 
    {
    	XStr str;
    	str<<proxyPrefix()<<type;
    	if (withTemplates) genTVars(str);
    	return str;
    }
    XStr baseName(int withTemplates=1) 
    {
    	XStr str;
    	str<<type->getBaseName();
    	if (withTemplates) genTVars(str);
    	return str;
    }
    int  isTemplated(void) { return (templat!=0); }
    int  isDerived(void) { return (bases!=0); }
    int  isAbstract(void) { return abstract; }
    virtual int isMainChare(void) {return 0;}
    virtual int isArray(void) {return 0;}
    virtual int isGroup(void) {return 0;}
    virtual int isNodeGroup(void) {return 0;}
    void setAbstract(int a) { abstract = a; }
    void print(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void genDecls(XStr &str);
    virtual void genSubDecls(XStr& str);
    virtual char *chareTypeName(void) {return "chare";}
    virtual char *proxyPrefix(void) {return "CProxy_";}
};

class MainChare : public Chare {
  public:
    MainChare(NamedType *t, TypeList *b=0, MemberList *l=0):Chare(t,b,l) {}
    virtual int isMainChare(void) {return 1;}
    virtual char *chareTypeName(void) {return "mainchare";}
};

class Array : public Chare {
  public:
    Array(NamedType *t, TypeList *b=0, MemberList *l=0):Chare(t,b,l) {}
    virtual int isArray(void) {return 1;}
    virtual void genSubDecls(XStr& str);
    virtual char *chareTypeName(void) {return "array";}
};

class Group : public Chare {
  public:
    Group(NamedType *t, TypeList *b=0, MemberList *l=0):Chare(t,b,l) {}
    virtual int isGroup(void) {return 1;}
    virtual void genSubDecls(XStr& str);
    virtual char *chareTypeName(void) {return "group";}
};

class NodeGroup : public Group {
  public:
    NodeGroup(NamedType *t, TypeList *b=0, MemberList *l=0):Group(t,b,l) {}
    virtual int isNodeGroup(void) {return 1;}
    virtual char *chareTypeName(void) {return "nodegroup";}
};


#define SPACKED  0x01
#define SVARSIZE 0x02

class Message : public TEntity, public Construct {
    int attrib;
    NamedType *type;
  public:
    Message(NamedType *t, int a) : attrib(a), type(t) { setTemplate(0); }
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    int  isPacked(void) { return attrib&SPACKED; }
    int  isVarsize(void) { return attrib&SVARSIZE; }
    virtual char *proxyPrefix(void) {return msg_prefix();}
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
#define SVIRTUAL  0x08
#define SPURE     0x10

class Entry : public Member {
  private:
    int attribs;
    EnType *retType;
    char *name;
    EnType *param;
    Value *stacksize;
    XStr epIdx(int include__idx_=1);
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
    void genGroupDefs(XStr& str);
    
    XStr paramType(void);
    XStr paramComma(void);
    XStr voidParamDecl(void);
    XStr callThread(const XStr &procName,int prependEntryName=0);
  public:
    Entry(int a, EnType *r, char *n, EnType *p, Value *sz=0) :
      attribs(a), retType(r), name(n), param(p), stacksize(sz)
    { setChare(0); 
      if(!isVirtual() && isPure()) {
        cerr << "Non-virtual methods cannot be pure virtual!!\n";
        abort();
      }
      if(!isThreaded() && stacksize) {
        cerr << "Non-Threaded methods cannot have stacksize spec.!!\n";
        abort();
      }
      if(retType && !isSync() && !retType->isVoid()) {
        cerr << "Async methods cannot have non-void return type!!\n";
        abort();
      }
    }
    int getStackSize(void) { return (stacksize ? stacksize->getIntVal() : 0); }
    int isThreaded(void) { return (attribs & STHREADED); }
    int isSync(void) { return (attribs & SSYNC); }
    int isConstructor(void) { return !strcmp(name, container->baseName(0).get_string());}
    int isExclusive(void) { return (attribs & SLOCKED); }
    int isVirtual(void) { return (attribs & SVIRTUAL); }
    const char *Virtual(void) {return isVirtual()?"virtual ":"";}
    int isPure(void) { return (attribs & SPURE); }
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
    void generate();
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
    void generate();
};

class Readonly : public Construct, public Member {
    int msg; // is it a readonly var(0) or msg(1) ?
    Type *type;
    char *name;
    ValueList *dims;
  public:
    Readonly(Type *t, char *n, ValueList* d, int m=0) : msg(m), type(t), name(n)
    { dims=d; setChare(0); }
    void print(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

#endif
