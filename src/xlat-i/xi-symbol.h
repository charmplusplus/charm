/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _SYMBOL_H
#define _SYMBOL_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <list>

#include "xi-util.h"
#include "EToken.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "CList.h"
#include "CStateVar.h"
#include "CParsedFile.h"

namespace xi {

/******************* Utilities ****************/

class Prefix {
public:
  static const char *Proxy;
  static const char *ProxyElement;
  static const char *ProxySection;
  static const char *Message;
  static const char *Index;
  static const char *Python;
};

typedef enum {
  forAll=0,forIndividual=1,forSection=2,forPython=3,forIndex=-1
} forWhom;

class Chare;//Forward declaration
class Message;
class TParamList;
extern int fortranMode;
extern int internalMode;
extern const char *cur_file;
void die(const char *why,int line=-1);

class Value : public Printable {
  private:
    int factor;
    const char *val;
  public:
    Value(const char *s);
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
    void printValue(XStr& str) {
      if(val) {
        val->print(str);
      }
      if(next) {
	  std::cout << "Unsupported type\n";
	  abort();
      }
    }
};

class Module;

class Construct : public Printable {
  protected:
    int external;
  public:
    int line;
    Module *containerModule;
    Construct() {external=0;line=-1;}
    void setExtern(int &e) { external = e; }
    void setModule(Module *m) { containerModule = m; }
    virtual void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) = 0;
    virtual void genDecls(XStr& str) = 0;
    virtual void genDefs(XStr& str) = 0;
    virtual void genReg(XStr& str) = 0;
    virtual void preprocess() { }

    // DMK - Accel Support
    virtual int genAccels_spe_c_funcBodies(XStr& str) = 0;
    virtual void genAccels_spe_c_regFuncs(XStr& str) = 0;
    virtual void genAccels_spe_c_callInits(XStr& str) = 0;
    virtual void genAccels_spe_h_includes(XStr& str) = 0;
    virtual void genAccels_spe_h_fiCountDefs(XStr& str) = 0;
    virtual void genAccels_ppe_c_regFuncs(XStr& str) = 0;
};

class ConstructList : public Construct {
    std::list<Construct*> constructs;
  public:
    ConstructList(int l, Construct *c, ConstructList *n=0);
    void setExtern(int e);
    void setModule(Module *m);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void preprocess();

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str);
    void genAccels_spe_h_fiCountDefs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);
};

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
       : name(n), tparams(t), scope(scope_) {}
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
    const char *getBaseName(void) const { return type->getBaseName(); }
    const char *getScope(void) const { return NULL; }
    virtual void genMsgProxyName(XStr& str) { 
      if(numstars != 1) {
        die("too many stars-- entry parameter must have form 'MTYPE *msg'"); 
      } else {
        type->genMsgProxyName(str);
      }
    }
};

/* I don't think these are useful any longer (OSL 11/30/2001) */
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

//This is used as a list of base classes
class TypeList : public Printable {
    Type *type;
    TypeList *next;
  public:
    TypeList(Type *t, TypeList *n=0) : type(t), next(n) {}
    int length(void) const;
    Type *getFirst(void) {return type;}
    Type *getSecond(void) {if (next) return next->getFirst(); return NULL;}
    void print(XStr& str);
    void genProxyNames(XStr& str, const char *prefix, const char *middle, 
                        const char *suffix, const char *sep, forWhom forElement);
};

/**************** Parameter types & lists (for marshalling) ************/
class Parameter {
    Type *type;
    const char *name; /*The name of the variable, if any*/
    const char *given_name; /*The name of the msg in ci file, if any*/
    const char *arrLen; /*The expression for the length of the array;
    			 NULL if not an array*/
    Value *val; /*Initial value, if any*/
    int line;
    int byReference; //Fake a pass-by-reference (for efficiency)
    int conditional; //If the parameter is conditionally packed

    // DMK - Added field for accelerator options
    int accelBufferType;
    XStr* accelInstName;

    friend class ParamList;
    void pup(XStr &str);
    void copyPtr(XStr &str);
    void marshallArraySizes(XStr &str);
    void marshallArrayData(XStr &str);
    void beginUnmarshall(XStr &str);
    void unmarshallArrayData(XStr &str);
    void pupAllValues(XStr &str);
  public:
    Parameter(int Nline,Type *Ntype,const char *Nname=0,
    	const char *NarrLen=0,Value *Nvalue=0);
    void setConditional(int c) { conditional = c; if (c) byReference = false; };
    void print(XStr &str,int withDefaultValues=0,int useConst=1);
    void printAddress(XStr &str);
    void printValue(XStr &str);
    int isMessage(void) const {return type->isMessage();}
    int isVoid(void) const {return type->isVoid();}
    int isCkArgMsgPtr(void) const {return type->isCkArgMsgPtr();}
    int isCkMigMsgPtr(void) const {return type->isCkMigMsgPtr();}
    int isArray(void) const {return arrLen!=NULL;}
    int isConditional(void) const {return conditional;}
    Type *getType(void) {return type;}
    const char *getArrayLen(void) const {return arrLen;}
    const char *getGivenName(void) const {return given_name;}
    const char *getName(void) const {return name;}
    void printMsg(XStr& str) {
      type->print(str);
      if(given_name!=0)
        str <<given_name;
    }
    int operator==(const Parameter &parm) const {
      return *type == *parm.type;
    }

    // DMK - Added for accelerator options
    public:
    enum {
      ACCEL_BUFFER_TYPE_UNKNOWN   = 0,
      ACCEL_BUFFER_TYPE_MIN       = 1,
      ACCEL_BUFFER_TYPE_READWRITE = 1,
      ACCEL_BUFFER_TYPE_READONLY  = 2,
      ACCEL_BUFFER_TYPE_WRITEONLY = 3,
      ACCEL_BUFFER_TYPE_MAX       = 3
    };
    void setAccelBufferType(int abt) {
      accelBufferType = ((abt < ACCEL_BUFFER_TYPE_MIN || abt > ACCEL_BUFFER_TYPE_MAX) ? (ACCEL_BUFFER_TYPE_UNKNOWN) : (abt));
    }
    int getAccelBufferType() { return accelBufferType; }
    void setAccelInstName(XStr* ain) { accelInstName = ain; }
    XStr* getAccelInstName(void) { return accelInstName; }

};
class ParamList {
    typedef int (Parameter::*pred_t)(void) const;
    int orEach(pred_t f);
    typedef void (Parameter::*fn_t)(XStr &str);
    void callEach(fn_t f,XStr &str);
    bool manyPointers;
  public:
    Parameter *param;
    ParamList *next;
    ParamList(ParamList *pl) :param(pl->param), next(pl->next), manyPointers(false) {}
    ParamList(Parameter *Nparam,ParamList *Nnext=NULL)
    	:param(Nparam), next(Nnext) { 
          manyPointers = false;
          if(next != NULL && (param->isMessage() || next->isMessage())){
            manyPointers = true;
          }
    }
    void print(XStr &str,int withDefaultValues=0,int useConst=1);
    void printAddress(XStr &str);
    void printValue(XStr &str);
    int isNamed(void) const {return param->type->isNamed();}
    int isBuiltin(void) const {return param->type->isBuiltin();}
    int isMessage(void) const {
    	return (next==NULL) && param->isMessage();
    }
    const char *getArrayLen(void) const {return param->getArrayLen();}
    int isArray(void) const {return param->isArray();}
    int isReference(void) const {return param->type->isReference();}
    int isVoid(void) const {
    	return (next==NULL) && param->isVoid();
    }
    int isPointer(void) const {return param->type->isPointer();}
    const char *getGivenName(void) const {return param->getGivenName();}
    int isMarshalled(void) const {
    	return !isVoid() && !isMessage();
    }
    int isCkArgMsgPtr(void) const {
        return (next==NULL) && param->isCkArgMsgPtr();
    }
    int isCkMigMsgPtr(void) const {
        return (next==NULL) && param->isCkMigMsgPtr();
    }
    int getNumStars(void) const {return param->type->getNumStars(); }
    const char *getBaseName(void) {
    	return param->type->getBaseName();
    }
    void genMsgProxyName(XStr &str) {
    	param->type->genMsgProxyName(str);
    }
    void printMsg(XStr& str) {
        ParamList *pl;
        param->printMsg(str);
        pl = next;
        while (pl != NULL)
        {
           str <<", ";
           pl->param->printMsg(str);
           pl = pl->next;
        } 
    }
    void preprocess();
    int hasConditional();
    void marshall(XStr &str, XStr &entry);
    void beginUnmarshall(XStr &str);
    void unmarshall(XStr &str, int isFirst=1);
    void unmarshallAddress(XStr &str, int isFirst=1);
    void pupAllValues(XStr &str);
    void endUnmarshall(XStr &str);
    int operator==(const ParamList &plist) const {
      if (!(*param == *(plist.param))) return 0;
      if (!next && !plist.next) return 1;
      if (!next || !plist.next) return 0;
      return *next ==  *plist.next;
    }
    void checkParamList();
};

class FuncType : public Type {
  private:
    Type *rtype;
    const char *name;
    ParamList *params;
  public:
    FuncType(Type* r, const char* n, ParamList* p)
    	:rtype(r),name(n),params(p) {}
    void print(XStr& str) { 
      rtype->print(str);
      str << "(*" << name << ")(";
      if(params)
        params->print(str);
    }
    const char *getBaseName(void) const { return name; }
    const char *getScope(void) const { return NULL; }
};

/****************** Template Support **************/
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
    std::string to_string();
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
    const char *val;
  public:
    TParamVal(const char *v) : val(v) {}
    void print(XStr& str) { str << val; }
    void genSpec(XStr& str) { str << val; }
};

class Scope : public Construct {
  protected:
    const char* name_;
    ConstructList* contents_;
  public:
    Scope(const char* name, ConstructList* contents) : contents_(contents), name_(name) {}
    virtual void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) {
        contents_->genPub(declstr, defstr, defconstr, connectPresent);
    }
    virtual void genDecls(XStr& str) {
        str << "namespace " << name_ << " {\n";
        contents_->genDecls(str);
        str << "} // namespace " << name_ << "\n";
    }
    virtual void genDefs(XStr& str) {
        str << "namespace " << name_ << " {\n";
        contents_->genDefs(str);
        str << "} // namespace " << name_ << "\n";
    }
    virtual void genReg(XStr& str) {
        str << "using namespace " << name_ << ";\n";
        contents_->genReg(str);
    }
    virtual void preprocess() {
        contents_->preprocess();
    }
    virtual void print(XStr& str) {
        str << "namespace " << name_ << "{\n";
        contents_->print(str);
        str << "} // namespace " << name_ << "\n";
    }

    // DMK - Accel Support
    virtual int genAccels_spe_c_funcBodies(XStr& str) {
      return contents_->genAccels_spe_c_funcBodies(str);
    }
    virtual void genAccels_spe_c_regFuncs(XStr& str) {
      contents_->genAccels_spe_c_regFuncs(str);
    }
    virtual void genAccels_spe_c_callInits(XStr& str) {
      contents_->genAccels_spe_c_callInits(str);
    }
    virtual void genAccels_spe_h_includes(XStr& str) {
      contents_->genAccels_spe_h_includes(str);
    }
    virtual void genAccels_spe_h_fiCountDefs(XStr& str) {
      contents_->genAccels_spe_h_fiCountDefs(str);
    }
    virtual void genAccels_ppe_c_regFuncs(XStr& str) {
      contents_->genAccels_ppe_c_regFuncs(str);
    }
};

class UsingScope : public Construct {
  protected:
    const char* name_;
    bool symbol_;
  public:
    UsingScope(const char* name, bool symbol=false) : name_(name), symbol_(symbol) {}
    virtual void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) {}
    virtual void genDecls(XStr& str) {
        str << "using ";
        if (!symbol_) str << "namespace ";
        str << name_ << ";\n";
    }
    virtual void genDefs(XStr& str) {}
    virtual void genReg(XStr& str) {}
    virtual void preprocess() {}
    virtual void print(XStr& str) {
        str << "using ";
        if (!symbol_) str << "namespace ";
        str << name_ << ";\n";
    }

    // DMK - Accel Support
    virtual int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    virtual void genAccels_spe_c_regFuncs(XStr& str) { }
    virtual void genAccels_spe_c_callInits(XStr& str) { }
    virtual void genAccels_spe_h_includes(XStr& str) { }
    virtual void genAccels_spe_h_fiCountDefs(XStr& str) { }
    virtual void genAccels_ppe_c_regFuncs(XStr& str) { }
};


/* A template construct */
class TVarList;
class TEntity;

class Template : public Construct {
    TVarList *tspec;
    TEntity *entity;
  public:
    Template(TVarList *t, TEntity *e) : tspec(t), entity(e) {}
    virtual void setExtern(int e);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void genSpec(XStr& str);
    void genVars(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str);
    void genAccels_spe_h_fiCountDefs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);
};

/* An entity that could be templated, i.e. chare, group or a message */
class TEntity : public Construct {
  protected:
    Template *templat;
  public:
    void setTemplate(Template *t) { templat = t; }
    virtual XStr tspec(void) { 
    	XStr str; 
    	if (templat) templat->genSpec(str); 
    	return str;
    }
    virtual XStr tvars(void) { 
    	XStr str;
    	if (templat) templat->genVars(str); 
    	return str;
    }
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
    const char *init;
  public:
    TFunc(FuncType *t, const char *v=0) : type(t), init(v) {}
    void print(XStr& str) { type->print(str); if(init) str << "=" << init; }
    void genLong(XStr& str){ type->print(str); if(init) str << "=" << init; }
    void genShort(XStr& str) {str << type->getBaseName(); }
};

/* A formal variable argument */
class TName : public TVar {
    Type *type;
    const char *name;
    const char *val;
  public:
    TName(Type *t, const char *n, const char *v=0) : type(t), name(n), val(v) {}
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

/******************* Chares, Arrays, Groups ***********/

/* Member of a chare or group, i.e. entry, RO or ROM */
class Member : public Construct {
   //friend class CParsedFile;
  protected:
    Chare *container;
  public:
    virtual void setChare(Chare *c) { container = c; }
    virtual int isSdag(void) { return 0; }
    virtual void collectSdagCode(CParsedFile *pf, int& sdagPresent) { return; }
    XStr makeDecl(const XStr &returnType,int forProxy=0);
    virtual void genPythonDecls(XStr& str) {}
    virtual void genIndexDecls(XStr& str)=0;
    virtual void genPythonDefs(XStr& str) {}
    virtual void genPythonStaticDefs(XStr& str) {}
    virtual void genPythonStaticDocs(XStr& str) {}
    virtual void lookforCEntry(CEntry *centry)  {}
};

/* List of members of a chare or group */
class MemberList : public Printable {
    std::list<Member*> members;
  public:
    MemberList(Member *m, MemberList *n=0);
    void appendMember(Member *m);
    void print(XStr& str);
    void setChare(Chare *c);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genPythonDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void preprocess();

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str);
    void genAccels_spe_h_fiCountDefs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);

    void genPythonDefs(XStr& str);
    void genPythonStaticDefs(XStr& str);
    void genPythonStaticDocs(XStr& str);
    void collectSdagCode(CParsedFile *pf, int& sdagPresent);
    virtual void lookforCEntry(CEntry *centry);
};

/* Chare or group is a templated entity */
class Chare : public TEntity {
  public:
    enum { //Set these attribute bits in "attrib"
    	CMIGRATABLE=1<<2,
	CPYTHON=1<<3,
    	CCHARE=1<<9,                 // plain non-migratable chare
    	CMAINCHARE=1<<10,
    	CARRAY=1<<11,
    	CGROUP=1<<12,
    	CNODEGROUP=1<<13
    };
    typedef unsigned int attrib_t;
  protected:
    attrib_t attrib;
    int hasElement;//0-- no element type; 1-- has element type
    forWhom forElement;
    int hasSection; //1-- applies only to array section

    NamedType *type;
    MemberList *list;
    TypeList *bases; //Base classes used by proxy
    TypeList *bases_CBase; //Base classes used by CBase (or NULL)
    
    int entryCount;
    int hasSdagEntry;

    void genTypedefs(XStr& str);
    void genRegisterMethodDef(XStr& str);
    void sharedDisambiguation(XStr &str,const XStr &superclass);
  public:
    Chare(int ln, attrib_t Nattr,
    	NamedType *t, TypeList *b=0, MemberList *l=0);
    void genProxyNames(XStr& str, const char *prefix, const char *middle, 
                        const char *suffix, const char *sep);
    void genIndexNames(XStr& str, const char *prefix, const char *middle, 
                        const char *suffix, const char *sep);
    XStr proxyName(int withTemplates=1); 
    XStr indexName(int withTemplates=1); 
    XStr indexList();
    XStr baseName(int withTemplates=1) 
    {
    	XStr str;
    	str<<type->getBaseName();
    	if (withTemplates) str<<tvars();
    	return str;
    }
    int  isTemplated(void) { return (templat!=0); }
    int  isMigratable(void) { return attrib&CMIGRATABLE; }
    int  isPython(void) { return attrib&CPYTHON; }
    int  isMainChare(void) {return attrib&CMAINCHARE;}
    int  isChare(void) {return attrib&CCHARE;}     // plain non-migratable chare
    int  isArray(void) {return attrib&CARRAY;}
    int  isGroup(void) {return attrib&CGROUP;}
    int  isNodeGroup(void) {return attrib&CNODEGROUP;}
    int  isForElement(void) const {return forElement==forIndividual;}
    int  isForSection(void) const {return forElement==forSection;}
    int  hasSdag() const { return hasSdagEntry; }
    void  setSdag(int f) { hasSdagEntry = f; }
    forWhom getForWhom(void) const {return forElement;}
    void print(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr &str);
    void preprocess();

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) {
      int rtn = 0;
      if (list) { rtn += list->genAccels_spe_c_funcBodies(str); }
      return rtn;
    }
    void genAccels_spe_c_regFuncs(XStr& str) {
      if (list) { list->genAccels_spe_c_regFuncs(str); }
    }
    void genAccels_spe_c_callInits(XStr& str) {
      if (list) { list->genAccels_spe_c_callInits(str); }
    }
    void genAccels_spe_h_includes(XStr& str) {
      if (list) { list->genAccels_spe_h_includes(str); }
    }
    void genAccels_spe_h_fiCountDefs(XStr& str) {
      if (list) { list->genAccels_spe_h_fiCountDefs(str); }
    }
    void genAccels_ppe_c_regFuncs(XStr& str) {
      if (list) { list->genAccels_ppe_c_regFuncs(str); }
    }

    int nextEntry(void) {return entryCount++;}
    virtual void genSubDecls(XStr& str);
    void genPythonDecls(XStr& str);
    void genPythonDefs(XStr& str);
    virtual char *chareTypeName(void) {return (char *)"chare";}
    virtual char *proxyPrefix(void);
    virtual void genSubRegisterMethodDef(XStr& str);
    void lookforCEntry(CEntry *centry);
};

class MainChare : public Chare {
  public:
    MainChare(int ln, attrib_t Nattr, 
    	NamedType *t, TypeList *b=0, MemberList *l=0):
	    Chare(ln, Nattr|CMAINCHARE, t,b,l) {}
    virtual char *chareTypeName(void) {return (char *) "mainchare";}
};

class Array : public Chare {
  protected:
    XStr indexSuffix;
    XStr indexType;//"CkArrayIndex"+indexSuffix;
  public:
    Array(int ln, attrib_t Nattr, NamedType *index,
    	NamedType *t, TypeList *b=0, MemberList *l=0);
    virtual int is1D(void) {return indexSuffix==(const char*)"1D";}
    virtual const char* dim(void) {return indexSuffix.get_string_const();}
    virtual void genSubDecls(XStr& str);
    virtual char *chareTypeName(void) {return (char *) "array";}
};

class Group : public Chare {
  public:
    Group(int ln, attrib_t Nattr,
    	NamedType *t, TypeList *b=0, MemberList *l=0);
    virtual void genSubDecls(XStr& str);
    virtual char *chareTypeName(void) {return (char *) "group";}
    virtual void genSubRegisterMethodDef(XStr& str);
};

class NodeGroup : public Group {
  public:
    NodeGroup(int ln, attrib_t Nattr,
    	NamedType *t, TypeList *b=0, MemberList *l=0):
	    Group(ln,Nattr|CNODEGROUP,t,b,l) {}
    virtual char *chareTypeName(void) {return (char *) "nodegroup";}
};


/****************** Messages ***************/
class Message; // forward declaration

class MsgVar {
 public:
  Type *type;
  const char *name;
  int cond;
  int array;
  MsgVar(Type *t, const char *n, int c, int a) : type(t), name(n), cond(c), array(a) { }
  Type *getType() { return type; }
  const char *getName() { return name; }
  int isConditional() { return cond; }
  int isArray() { return array; }
  void print(XStr &str) {str<<(isConditional()?"conditional ":"");type->print(str);str<<" "<<name<<(isArray()?"[]":"")<<";";}
};

class MsgVarList : public Printable {
 public:
  MsgVar *msg_var;
  MsgVarList *next;
  MsgVarList(MsgVar *mv, MsgVarList *n=0) : msg_var(mv), next(n) {}
  void print(XStr &str) {
    msg_var->print(str);
    str<<"\n";
    if(next) next->print(str);
  }
  int len(void) { return (next==0)?1:(next->len()+1); }
};

class Message : public TEntity {
    NamedType *type;
    MsgVarList *mvlist;
    void printVars(XStr& str) {
      if(mvlist!=0) {
        str << "{\n";
        mvlist->print(str);
        str << "}\n";
      }
    }
  public:
    Message(int l, NamedType *t, MsgVarList *mv=0)
      : type(t), mvlist(mv) 
      { line=l; setTemplate(0); }
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) {}
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    void genAccels_spe_c_regFuncs(XStr& str) { }
    void genAccels_spe_c_callInits(XStr& str) { }
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str) { }

    virtual const char *proxyPrefix(void) {return Prefix::Message;}
    void genAllocDecl(XStr& str);
    int numArrays(void) {
      if (mvlist==0) return 0;
      int count = 0;
      MsgVarList *mv = mvlist;
      for (int i=0; i<mvlist->len(); ++i, mv=mv->next) if (mv->msg_var->isArray()) count ++;
      return count;
    }
    int numConditional(void) {
      if (mvlist==0) return 0;
      int count = 0;
      MsgVarList *mv = mvlist;
      for (int i=0; i<mvlist->len(); ++i, mv=mv->next) if (mv->msg_var->isConditional()) count ++;
      return count;
    }
    int numVars(void) { return ((mvlist==0) ? 0 : mvlist->len()); }
};





/******************* Entry Point ****************/
// Entry attributes
#define STHREADED 0x01
#define SSYNC     0x02
#define SLOCKED   0x04
#define SPURE     0x10
#define SMIGRATE  0x20 //<- is magic migration constructor
#define SCREATEHERE   0x40 //<- is a create-here-if-nonexistant
#define SCREATEHOME   0x80 //<- is a create-at-home-if-nonexistant
#define SIMMEDIATE    0x100 //<- is a immediate
#define SNOKEEP       0x200
#define SNOTRACE      0x400
#define SSKIPSCHED    0x800 //<- is a message skipping charm scheduler
#define SPYTHON       0x1000
#define SINLINE       0x2000 //<- inline message
#define SIGET   0x4000 
#define SLOCAL        0x8000 //<- local message
#define SACCEL  0x10000
#define SMEM  0x20000

/* An entry construct */
class Entry : public Member {
  private:
    int line,entryCount;
    int attribs;    
    Type *retType;
    Value *stacksize;
    const char *pythonDoc;
    
    XStr proxyName(void) {return container->proxyName();}
    XStr indexName(void) {return container->indexName();}

//    friend class CParsedFile;
    int hasCallMarshall;
    void genCall(XStr &dest,const XStr &preCall);

    XStr epStr(void);
    XStr epIdx(int fromProxy=1);
    XStr chareIdx(int fromProxy=1);
    void genEpIdxDecl(XStr& str);
    void genEpIdxDef(XStr& str);
    
    void genChareDecl(XStr& str);
    void genChareStaticConstructorDecl(XStr& str);
    void genChareStaticConstructorDefs(XStr& str);
    void genChareDefs(XStr& str);
    
    void genArrayDefs(XStr& str);
    void genArrayStaticConstructorDecl(XStr& str);
    void genArrayStaticConstructorDefs(XStr& str);
    void genArrayDecl(XStr& str);
    
    void genGroupDecl(XStr& str);
    void genGroupStaticConstructorDecl(XStr& str);
    void genGroupStaticConstructorDefs(XStr& str);
    void genGroupDefs(XStr& str);
    
    void genPythonDecls(XStr& str);
    void genPythonDefs(XStr& str);
    void genPythonStaticDefs(XStr& str);
    void genPythonStaticDocs(XStr& str);

    // DMK - Accel Support
    void genAccelFullParamList(XStr& str, int makeRefs);
    void genAccelFullCallList(XStr& str);
    void genAccelIndexWrapperDecl_general(XStr& str);
    void genAccelIndexWrapperDef_general(XStr& str);
    void genAccelIndexWrapperDecl_spe(XStr& str);
    void genAccelIndexWrapperDef_spe(XStr& str);
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str) { }
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str);

    XStr paramType(int withDefaultVals,int withEO=0,int useConst=1);
    XStr paramComma(int withDefaultVals,int withEO=0);
    XStr eo(int withDefaultVals,int priorComma=1);
    XStr syncReturn(void);
    XStr marshallMsg(void);
    XStr callThread(const XStr &procName,int prependEntryName=0);
  public:
    SdagConstruct *sdagCon;
    TList<CStateVar *> *stateVars;
    TList<CStateVar *> *stateVarsChildren;
    TList<CStateVar *> estateVars;
    CEntry *entryPtr;
    XStr *label;
    char *name;
    const char *intExpr;
    ParamList *param;
    ParamList *connectParam;
    int isConnect;
    int isWhenEntry;

    // DMK - Accel Support
    ParamList* accelParam;
    XStr* accelCodeBody;
    XStr* accelCallbackName;
    void setAccelParam(ParamList* apl) { accelParam = apl; }
    void setAccelCodeBody(XStr* acb) { accelCodeBody = acb; }
    void setAccelCallbackName(XStr* acbn) { accelCallbackName = acbn; }

    // DMK - Accel Support
    int accel_numScalars;
    int accel_numArrays;
    int accel_dmaList_numReadOnly;
    int accel_dmaList_numReadWrite;
    int accel_dmaList_numWriteOnly;
    int accel_dmaList_scalarNeedsWrite;

    Entry(int l, int a, Type *r, const char *n, ParamList *p, Value *sz=0, SdagConstruct *sc =0, const char *e=0, int connect=0, ParamList *connectPList =0);
    void setChare(Chare *c);
    int isConnectEntry(void) { return isConnect; }
    int paramIsMarshalled(void) { return param->isMarshalled(); }
    int getStackSize(void) { return (stacksize ? stacksize->getIntVal() : 0); }
    int isThreaded(void) { return (attribs & STHREADED); }
    int isSync(void) { return (attribs & SSYNC); }
    int isIget(void) { return (attribs & SIGET); }
    int isConstructor(void) { return !strcmp(name, container->baseName(0).get_string());}
    int isExclusive(void) { return (attribs & SLOCKED); }
    int isImmediate(void) { return (attribs & SIMMEDIATE); }
    int isSkipscheduler(void) { return (attribs & SSKIPSCHED); }
    int isInline(void) { return attribs & SINLINE; }
    int isLocal(void) { return attribs & SLOCAL; }
    int isCreate(void) { return (attribs & SCREATEHERE)||(attribs & SCREATEHOME); }
    int isCreateHome(void) { return (attribs & SCREATEHOME); }
    int isCreateHere(void) { return (attribs & SCREATEHERE); }
    int isPython(void) { return (attribs & SPYTHON); }
    int isNoTrace(void) { return (attribs & SNOTRACE); }
    int isNoKeep(void) { return (attribs & SNOKEEP); }
    int isSdag(void) { return (sdagCon!=0); }

    // DMK - Accel support
    int isAccel(void) { return (attribs & SACCEL); }

    int isMemCritical(void) { return (attribs & SMEM); }

    void print(XStr& str);
    void genIndexDecls(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void preprocess();
    char *getEntryName() { return name; }
    void generateEntryList(TList<CEntry*>&, SdagConstruct *);
    void collectSdagCode(CParsedFile *pf, int& sdagPresent);
    void propagateState(int);
    void lookforCEntry(CEntry *centry);
};

class EntryList {
  public:
    Entry *entry;
    EntryList *next;
    EntryList(Entry *e,EntryList *elist=NULL):
    	entry(e), next(elist) {}
    void generateEntryList(TList<CEntry*>&, SdagConstruct *);
};


/******************** AccelBlock : Block of code for accelerator **********************/
class AccelBlock : public Construct {

 protected:

  XStr* code;

 private:

  void outputCode(XStr& str) {
    if (code != NULL) {
      str << "\n#ifndef CK_TEMPLATES_ONLY\n"
          << "/***** Accel_Block Start *****/\n"
          << (*(code))
          << "\n/***** Accel_Block End *****/\n"
          << "#endif /*CK_TEMPLATES_ONLY*/\n\n";
    }
  }

 public:

  /// Constructor(s)/Destructor ///
  AccelBlock(int l, XStr* c) { line = l; code = c; }
  ~AccelBlock() { delete code; }

  /// Printable Methods ///
  void print(XStr& str) { }

  /// Construct Methods ///
  void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) { }
  void genDecls(XStr& str) { }
  void genDefs(XStr& str) { outputCode(str); }
  void genReg(XStr& str) { }
  void preprocess() { }

  /// Construct Accel Support Methods ///
  int genAccels_spe_c_funcBodies(XStr& str) { outputCode(str); return 0; }
  void genAccels_spe_c_regFuncs(XStr& str) { }
  void genAccels_spe_c_callInits(XStr& str) { }
  void genAccels_spe_h_includes(XStr& str) { }
  void genAccels_spe_h_fiCountDefs(XStr& str) { }
  void genAccels_ppe_c_regFuncs(XStr& str) { }
};


/****************** Modules, etc. ****************/
class Module : public Construct {
    int _isMain;
    const char *name;
    
  public:
    ConstructList *clist;
    Module(int l, const char *n, ConstructList *c) : name(n), clist(c) {
	    line = l;
	    _isMain=0;
	    if (clist!=NULL) clist->setModule(this);
    }
    void print(XStr& str);
    void generate();
    void setModule();
    void prependConstruct(Construct *c) { clist = new ConstructList(-1, c, clist); }
    void preprocess();
    void genDepend(const char *cifile);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void setMain(void) { _isMain = 1; }
    int isMain(void) { return _isMain; }

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str);
    void genAccels_spe_h_fiCountDefs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);
};

class ModuleList : public Printable {
    std::list<Module*> modules;
  public:
    int line;
    ModuleList(int l, Module *m, ModuleList *n=0) : line(l)
	{
	    modules.push_back(m);
	    if (n)
		modules.insert(modules.end(),
			       n->modules.begin(), n->modules.end());
	}
    void print(XStr& str);
    void generate();
    void preprocess();
    void genDepends(std::string ciFileBaseName);
};

class Readonly : public Member {
    int msg; // is it a readonly var(0) or msg(1) ?
    Type *type;
    const char *name;
    ValueList *dims;
    XStr qName(void) const { /*Return fully qualified name*/
      XStr ret;
      if(container) ret<<container->baseName()<<"::";
      ret<<name;
      return ret;
    }
  public:
    Readonly(int l, Type *t, const char *n, ValueList* d, int m=0)
	    : msg(m), type(t), name(n)
            { line=l; dims=d; setChare(0); }
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent) {}
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    void genAccels_spe_c_regFuncs(XStr& str) { }
    void genAccels_spe_c_callInits(XStr& str) { }
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str) { }
};

class InitCall : public Member {

    const char *name; //Name of subroutine to call
    int isNodeCall;

    // DMK - Accel Support
    int isAccelFlag;

public:

    InitCall(int l, const char *n, int nodeCall);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    void genAccels_spe_c_regFuncs(XStr& str) { }
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str) { }

    void setAccel() { isAccelFlag = 1; }
    void clearAccel() { isAccelFlag = 0; }
    int isAccel() { return isAccelFlag; }
};

class PUPableClass : public Member {
    NamedType* type;
    PUPableClass *next; //Linked-list of PUPable classes
public:
    PUPableClass(int l, NamedType* type_, PUPableClass *next_);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) {
      int rtn=0;
      if (next) { rtn += next->genAccels_spe_c_funcBodies(str); }
      return rtn;
    }
    void genAccels_spe_c_regFuncs(XStr& str) {
      if (next) { next->genAccels_spe_c_regFuncs(str); }
    }
    void genAccels_spe_c_callInits(XStr& str) {
      if (next) { next->genAccels_spe_c_callInits(str); }
    }
    void genAccels_spe_h_includes(XStr& str) {
      if (next) { next->genAccels_spe_h_includes(str); }
    }
    void genAccels_spe_h_fiCountDefs(XStr& str) {
      if (next) { next->genAccels_spe_h_fiCountDefs(str); }
    }
    void genAccels_ppe_c_regFuncs(XStr& str) {
      if (next) { next->genAccels_ppe_c_regFuncs(str); }
    }
};

class IncludeFile : public Member {
    const char *name; //Name of include file
public:
    IncludeFile(int l, const char *name_);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    void genAccels_spe_c_regFuncs(XStr& str) { }
    void genAccels_spe_c_callInits(XStr& str) { }
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str) { }
};

class ClassDeclaration : public Member {
    const char *name; //Name of class 
public:
    ClassDeclaration(int l, const char *name_);
    void print(XStr& str);
    void genPub(XStr& declstr, XStr& defstr, XStr& defconstr, int& connectPresent);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str) { return 0; }
    void genAccels_spe_c_regFuncs(XStr& str) { }
    void genAccels_spe_c_callInits(XStr& str) { }
    void genAccels_spe_h_includes(XStr& str) { }
    void genAccels_spe_h_fiCountDefs(XStr& str) { }
    void genAccels_ppe_c_regFuncs(XStr& str) { }
};


/******************* Structured Dagger Constructs ***************/
class SdagConstruct { 
private:
  void generateWhen(XStr& op);
  void generateOverlap(XStr& op);
  void generateWhile(XStr& op);
  void generateFor(XStr& op);
  void generateIf(XStr& op);
  void generateElse(XStr& op);
  void generateForall(XStr& op);
  void generateOlist(XStr& op);
  void generateSdagEntry(XStr& op);
  void generateSlist(XStr& op);
  void generateAtomic(XStr& op);
  void generateForward(XStr& op);
  void generateConnect(XStr& op);
  void generatePrototype(XStr& op, ParamList *list);
  void generatePrototype(XStr& op, TList<CStateVar*>&);
  void generateCall(XStr& op, TList<CStateVar*>&);

  void generateTraceBeginCall(XStr& op);          // for trace
  void generateBeginTime(XStr& op);               //for Event Bracket
  void generateEventBracket(XStr& op, int eventType);     //for Event Bracket
  void generateListEventBracket(XStr& op, int eventType);
public:
  int nodeNum;
  XStr *label;
  XStr *counter;
  EToken type;
  char nameStr[128];
  XStr *traceName;	
  TList<SdagConstruct *> *constructs;
  TList<SdagConstruct *> *publishesList;
  TList<CStateVar *> *stateVars;
  TList<CStateVar *> *stateVarsChildren;
  SdagConstruct *next;
  ParamList *param;
  XStr *text;
  XStr *connectEntry;
  int nextBeginOrEnd;
  EntryList *elist;
  SdagConstruct *con1, *con2, *con3, *con4;
  SdagConstruct(EToken t, SdagConstruct *construct1);

  SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList);

  SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
              SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el);

  SdagConstruct(EToken t, const char *str) : type(t), con1(0), con2(0), con3(0), con4(0)
		{ text = new XStr(str); constructs = new TList<SdagConstruct*>(); 
                  publishesList = new TList<SdagConstruct*>(); }
                                             
 
  SdagConstruct(EToken t) : type(t), traceName(NULL), con1(0), con2(0), con3(0), con4(0) 
		{ publishesList = new TList<SdagConstruct*>();
		  constructs = new TList<SdagConstruct*>(); }

  SdagConstruct(EToken t, XStr *txt) : type(t), traceName(NULL), text(txt), con1(0), con2(0), con3(0), con4(0) 
                { publishesList = new TList<SdagConstruct*>();
		  constructs = new TList<SdagConstruct*>();  }
  SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl);
  void numberNodes(void);
  void labelNodes(void);
  void generateConnectEntryList(TList<SdagConstruct*>&);
  void generateConnectEntries(XStr&);
  void generateEntryList(TList<CEntry*>&, SdagConstruct *);
  void propagateState(int);
  void propagateState(TList<CStateVar*>&, TList<CStateVar*>&, TList<SdagConstruct*>&, int);
  void generateCode(XStr& output);
  void setNext(SdagConstruct *, int);

  // for trace
  void generateTrace();          
  void generateRegisterEp(XStr& output);          
  void generateTraceEpDecl(XStr& output);         
  void generateTraceEpDef(XStr& output);          
  static void generateTraceEndCall(XStr& op);            
  static void generateTlineEndCall(XStr& op);
  static void generateBeginExec(XStr& op, const char *name);
  static void generateEndExec(XStr& op);
  static void generateEndSeq(XStr& op);
  static void generateDummyBeginExecute(XStr& op);

};

extern void RemoveSdagComments(char *);

}

#endif
