#ifndef _SDAG_CONSTRUCT_H
#define _SDAG_CONSTRUCT_H

#include <list>

#include "xi-util.h"
#include "sdag-globals.h"

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
//
// TODO(Ralf): Find a good place to put these functions,
//             which are used by all constructs.
extern void generateClosureSignature(XStr& decls, XStr& defs,
                                     const Chare* chare, bool declareStatic,
                                     const char* returnType, const XStr* name,
                                     bool isEnd, std::list<EncapState*> params,
                                     int numRefs = 0);
extern void generateClosureSignature(XStr& decls, XStr& defs,
                                     const Entry* entry, bool declareStatic,
                                     const char* returnType, const XStr* name,
                                     bool isEnd, std::list<EncapState*> params,
                                     int numRefs = 0);
extern void endMethod(XStr& op);

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
  const char *label_str;

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
  virtual void numberNodes();
  virtual void labelNodes();
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

}   // namespace xi

#endif // ifndef _SDAG_CONSTRUCT_H
