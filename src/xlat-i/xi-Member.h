#ifndef _MEMBER_H
#define _MEMBER_H

#include "xi-Construct.h"

namespace xi {

class SdagCollection;
class ValueList;
class WhenStatementEChecker;
class Chare;
class CEntry;
class Type;
class NamedType;

/* Member of a chare or group, i.e. entry, RO or ROM */
class Member : public Construct {
 //friend class CParsedFile;
 protected:
  Chare *container;

 public:
  TVarList *tspec;
  Member() : container(0), tspec(0) { }
  inline Chare *getContainer() const { return container; }
  virtual void setChare(Chare *c) { container = c; }
  virtual void preprocessSDAG() { }
  virtual int isSdag(void) { return 0; }
  virtual void collectSdagCode(SdagCollection *) { return; }
  virtual void collectSdagCode(WhenStatementEChecker *) { return; }
  XStr makeDecl(const XStr &returnType,int forProxy=0, bool isStatic = false);
  virtual void genPythonDecls(XStr& ) {}
  virtual void genIndexDecls(XStr& ) {}
  virtual void genPythonDefs(XStr& ) {}
  virtual void genPythonStaticDefs(XStr&) {}
  virtual void genPythonStaticDocs(XStr&) {}
  virtual void lookforCEntry(CEntry *)  {}
  virtual void genTramTypes() {}
};

class Readonly : public Member {
    int msg; // is it a readonly var(0) or msg(1) ?
    Type *type;
    const char *name;
    ValueList *dims;
    XStr qName(void) const;
  public:
    Readonly(int l, Type *t, const char *n, ValueList* d, int m=0);
    void print(XStr& str);
    void genDecls(XStr& str);
    void genIndexDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
};

class InitCall : public Member {

    const char *name; //Name of subroutine to call
    int isNodeCall;

    // DMK - Accel Support
    int isAccelFlag;

public:

    InitCall(int l, const char *n, int nodeCall);
    void print(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    void genAccels_spe_c_callInits(XStr& str);

    void setAccel();
    void clearAccel();
    int isAccel();
};

class PUPableClass : public Member {
    NamedType* type;
    PUPableClass *next; //Linked-list of PUPable classes
public:
    PUPableClass(int l, NamedType* type_, PUPableClass *next_);
    void print(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);

    // DMK - Accel Support
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_spe_c_callInits(XStr& str);
    void genAccels_spe_h_includes(XStr& str);
    void genAccels_spe_h_fiCountDefs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);
};

class IncludeFile : public Member {
    const char *name; //Name of include file
public:
    IncludeFile(int l, const char *name_);
    void print(XStr& str);
    void genDecls(XStr& str);
};

class ClassDeclaration : public Member {
    const char *name; //Name of class 
public:
    ClassDeclaration(int l, const char *name_);
    void print(XStr& str);
    void genDecls(XStr& str);
};


}   // namespace xi

#endif  // ifndef _MEMBER_H
