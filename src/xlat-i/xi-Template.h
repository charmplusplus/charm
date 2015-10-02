#ifndef _TEMPLATE_H
#define _TEMPLATE_H

#include "xi-Type.h"
#include "xi-Construct.h"

namespace xi {

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
  TParamList(TParam *t, TParamList *n=0);
  void print(XStr& str);
  void genSpec(XStr& str);
  std::string to_string();
};

/* A type instantiation parameter */
class TParamType : public TParam {
  Type *type;
 public:
  TParamType(Type *t);
  void print(XStr& str);
  void genSpec(XStr& str);
};

/* A Value instantiation parameter */
class TParamVal : public TParam {
    const char *val;
  public:
    TParamVal(const char *v);
    void print(XStr& str);
    void genSpec(XStr& str);
};

class Scope : public ConstructList {
  protected:
    const char* name_;
  public:
    Scope(const char* name, ConstructList* contents);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    void genGlobalCode(XStr scope, XStr &decls, XStr &defs);
    void print(XStr& str);
    void outputClosuresDecl(XStr& str);
    void outputClosuresDef(XStr& str);
};

class UsingScope : public Construct {
  protected:
    const char* name_;
    bool symbol_;
  public:
    UsingScope(const char* name, bool symbol=false);
    virtual void genDecls(XStr& str);
    virtual void print(XStr& str);
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
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genSpec(XStr& str, bool printDefault = true);
    void genVars(XStr& str);
    void genGlobalCode(XStr scope, XStr &decls, XStr &defs);
    void outputClosuresDecl(XStr& str);
    void outputClosuresDef(XStr& str);
    void preprocess();
    void check();

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
    void setTemplate(Template *t);
    virtual XStr tspec(bool printDefault = true) const;
    virtual XStr tvars(void) const;
};
/* A formal argument of a template */
class TVar : public Printable {
  public:
    virtual void genLong(XStr& str, bool printDefault = true) = 0;
    virtual void genShort(XStr& str) = 0;
};

/* a formal type argument */
class TType : public TVar {
    Type *type;
    Type *init;
  public:
    TType(Type *t, Type *i=0);
    void print(XStr& str);
    void genLong(XStr& str, bool printDefault = true);
    void genShort(XStr& str);
};

/* a formal function argument */
class TFunc : public TVar {
    FuncType *type;
    const char *init;
  public:
    TFunc(FuncType *t, const char *v=0);
    void print(XStr& str);
    void genLong(XStr& str, bool printDefault = true);
    void genShort(XStr& str);
};

/* A formal variable argument */
class TName : public TVar {
    Type *type;
    const char *name;
    const char *val;
  public:
    TName(Type *t, const char *n, const char *v=0);
    void print(XStr& str);
    void genLong(XStr& str, bool printDefault = true);
    void genShort(XStr& str);
};

/* A list of formal arguments to a template */
class TVarList : public Printable {
    TVar *tvar;
    TVarList *next;
  public:
    TVarList(TVar *v, TVarList *n=0);
    void print(XStr& str);
    void genLong(XStr& str, bool printDefault = true);
    void genShort(XStr& str);
};

}   // namespace xi

#endif  // ifndef _TEMPLATE_H
