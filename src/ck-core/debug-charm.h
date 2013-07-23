/*
 Interface to Charm++ portion of parallel debugger.
 Orion Sky Lawlor, olawlor@acm.org, 7/30/2001
 */
#ifndef __CMK_DEBUG_CHARM_H
#define __CMK_DEBUG_CHARM_H

#ifndef __cplusplus
#  error "debug-charm.h is for C++; use debug-conv.h for C programs"
#endif

#include "converse.h"
#include "debug-conv++.h"
#include "pup.h"
#include "cklists.h"

void CkPupMessage(PUP::er &p,void **atMsg,int pack_detail=1);

void *CpdGetCurrentObject();
void *CpdGetCurrentMsg();

//Hooks inside the debugger before and after an entry method is invoked
extern void CpdBeforeEp(int, void*, void*);
extern void CpdAfterEp(int);
extern void CpdFinishInitialization();

class CpdPersistentChecker {
public:
  virtual ~CpdPersistentChecker() {}
  virtual void cpdCheck(void*) {}
};

typedef struct DebugPersistentCheck {
  CpdPersistentChecker *object;
  void *msg;
  
  DebugPersistentCheck() : object(NULL), msg(NULL) {}
  DebugPersistentCheck(CpdPersistentChecker *o, void *m) : object(o), msg(m) {}
} DebugPersistentCheck;

// This class is the parallel of EntryInfo declared in register.h and is used
// to extend the former with additional debug information. There is a direct
// correspondence between elements on the two arrays.
class DebugEntryInfo {
public:
  // true if this entry method has a breakpoint set
  bool isBreakpoint;
  CkVec<DebugPersistentCheck> preProcess;
  CkVec<DebugPersistentCheck> postProcess;

  DebugEntryInfo() : isBreakpoint(false) { }
};

typedef CkVec<DebugEntryInfo> DebugEntryTable;

//These pup functions are useful in CpdLists, as they document the name
//  of the variable.  Your object must be named "c" (a stupid hack).
#define PCOM(field) p.comment(#field); p(c->field);
#define PCOMS(field) \
  if (!p.isUnpacking()) { \
  	p.comment(#field); p((char *)c->field,strlen(c->field)); \
  }

#endif
