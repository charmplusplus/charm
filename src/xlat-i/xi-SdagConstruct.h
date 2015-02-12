#ifndef _SDAG_CONSTRUCT_H
#define _SDAG_CONSTRUCT_H

#include <list>

#include "xi-util.h"

#include "EToken.h"

namespace xi {

class Entry;
class EntryList;
class EncapState;
class WhenConstruct;
class CStateVar;
class ParamList;
class CEntry;
class Chare;

/******************* Structured Dagger Constructs ***************/
class SdagConstruct { 
 private:
  void generateOverlap(XStr& decls, XStr& defs, Entry* entry);
  void generateWhile(XStr& decls, XStr& defs, Entry* entry);
  void generateFor(XStr& decls, XStr& defs, Entry* entry);
  void generateIf(XStr& decls, XStr& defs, Entry* entry);
  void generateElse(XStr& decls, XStr& defs, Entry* entry);
  void generateForall(XStr& decls, XStr& defs, Entry* entry);
  void generateOlist(XStr& decls, XStr& defs, Entry* entry);
  void generateSdagEntry(XStr& decls, XStr& defs, Entry *entry);
  void generateSlist(XStr& decls, XStr& defs, Entry* entry);
  void generateCaseList(XStr& decls, XStr& defs, Entry* entry);

 protected:
  void generateCall(XStr& op, std::list<EncapState*>& cur,
                    std::list<EncapState*>& next, const XStr* name,
                    const char* nameSuffix = 0);
  void generateTraceBeginCall(XStr& defs, int indent);          // for trace
  void generateBeginTime(XStr& defs);               //for Event Bracket
  void generateEventBracket(XStr& defs, int eventType);     //for Event Bracket
  void generateListEventBracket(XStr& defs, int eventType);
  void generateChildrenCode(XStr& decls, XStr& defs, Entry* entry);
  void generateChildrenEntryList(std::list<CEntry*>& CEntrylist, WhenConstruct *thisWhen);
  void propagateStateToChildren(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  std::list<SdagConstruct *> *constructs;
  std::list<CStateVar *> *stateVars;
  std::list<EncapState*> encapState, encapStateChild;
  std::list<CStateVar *> *stateVarsChildren;

 public:
  int unravelClosuresBegin(XStr& defs, bool child = false);
  void unravelClosuresEnd(XStr& defs, bool child = false);

  int nodeNum;
  XStr *label;
  XStr *counter;
  EToken type;
  char nameStr[128];
  XStr *traceName;	
  SdagConstruct *next;
  ParamList *param;
  //cppcheck-suppress unsafeClassCanLeak
  XStr *text;
  int nextBeginOrEnd;
  EntryList *elist;
  Entry* entry;
  SdagConstruct *con1, *con2, *con3, *con4;
  SdagConstruct(EToken t, SdagConstruct *construct1);

  SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList);

  SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
              SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el);

  SdagConstruct(EToken t, const char *str);
  SdagConstruct(EToken t);
  SdagConstruct(EToken t, XStr *txt);

  virtual ~SdagConstruct();

  void init(EToken& t);
  SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl);
  void numberNodes(void);
  void labelNodes();
  XStr* createLabel(const char* str, int nodeNum);
  virtual void generateEntryList(std::list<CEntry*>&, WhenConstruct *);
  void propagateState(int);
  virtual void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  virtual void generateCode(XStr& decls, XStr& defs, Entry *entry);
  void setNext(SdagConstruct *, int);
  void buildTypes(std::list<EncapState*>& state);

  // for trace
  virtual void generateTrace();
  void generateRegisterEp(XStr& defs);
  void generateTraceEp(XStr& decls, XStr& defs, Chare* chare);
  static void generateTraceEndCall(XStr& defs, int indent);
  static void generateTlineEndCall(XStr& defs);
  static void generateBeginExec(XStr& defs, const char *name);
  static void generateEndExec(XStr& defs);
  static void generateEndSeq(XStr& defs);
  static void generateDummyBeginExecute(XStr& defs, int indent);
};

/***************** WhenConstruct **************/
class WhenConstruct : public SdagConstruct {
 public:
  CStateVar* speculativeState;
  void generateCode(XStr& decls, XStr& defs, Entry *entry);
  WhenConstruct(EntryList *el, SdagConstruct *body);
  void generateEntryList(std::list<CEntry*>& CEntrylist, WhenConstruct *thisWhen);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateEntryName(XStr& defs, Entry* e, int curEntry);
  void generateWhenCode(XStr& op, int indent);
};

/***************** AtomicConstruct **************/
class AtomicConstruct : public SdagConstruct {
 public:
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
  void generateTrace();
  AtomicConstruct(const char *code, const char *trace_name);
};

/***************** WhileConstruct **************/
class WhileConstruct : public SdagConstruct {
 public:
  WhileConstruct(SdagConstruct *pred, SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** IfConstruct **************/
class IfConstruct : public SdagConstruct {
 public:
  IfConstruct(SdagConstruct *pred, SdagConstruct *then_body, SdagConstruct *else_body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** ForConstruct **************/
class ForConstruct : public SdagConstruct {
 public:
  ForConstruct(SdagConstruct *decl, SdagConstruct *pred, SdagConstruct *advance, SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** ForallConstruct **************/
class ForallConstruct : public SdagConstruct {
 public:
  ForallConstruct(SdagConstruct *tag, SdagConstruct *begin, SdagConstruct *end, SdagConstruct *step, SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** CaseConstruct **************/
class CaseConstruct : public SdagConstruct {
 public:
  CaseConstruct(SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

}   // namespace xi

#endif // ifndef _SDAG_CONSTRUCT_H
