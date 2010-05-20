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
#include "debug-conv.h"
#include "pup.h"
#include "cklists.h"

#define CHARMDEBUG_MAJOR   10
#define CHARMDEBUG_MINOR    0

void *CpdGetCurrentObject();
void *CpdGetCurrentMsg();

//Hooks inside the debugger before and after an entry method is invoked
extern void CpdBeforeEp(int, void*, void*);
extern void CpdAfterEp(int);
extern void CpdFinishInitialization();

class CpdPersistentChecker {
public:
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
  CmiBool isBreakpoint;
  CkVec<DebugPersistentCheck> preProcess;
  CkVec<DebugPersistentCheck> postProcess;

  DebugEntryInfo() : isBreakpoint(CmiFalse) { }
};

typedef CkVec<DebugEntryInfo> DebugEntryTable;

//These pup functions are useful in CpdLists, as they document the name
//  of the variable.  Your object must be named "c" (a stupid hack).
#define PCOM(field) p.comment(#field); p(c->field);
#define PCOMS(field) \
  if (!p.isUnpacking()) { \
  	p.comment(#field); p((char *)c->field,strlen(c->field)); \
  }

/**
  A CpdListAccessor responds to CCS requests for a single CpdList.
  To make some data available via the CpdList interface, you 
  make a subclass of this class (possibly CpdSimpleListAccessor),
  and pass it to CpdListRegister, below.
*/
class CpdListAccessor {
protected:
  /**
    Subclasses must call this before pupping each requested item.
    This inserts a marker to allow the client to distinguish between
    different CpdList items. 
  */
  void beginItem(PUP::er &p,int itemNo);
  /// Decides if this CpdList requires boundary checking
  bool checkBound;
public:
  CpdListAccessor() : checkBound(true) {}
  virtual ~CpdListAccessor(); 
  /// Return the CpdList path CCS clients should use to access this data.
  virtual const char *getPath(void) const =0;
  /// Return the length of this CpdList.
  virtual size_t getLength(void) const =0;
  /// Does this CpdList requires boundary checking?
  virtual bool checkBoundary(void) const { return checkBound; }
  /// Pup the items listed in this request.  Be sure to call beginItem between items!
  virtual void pup(PUP::er &p,CpdListItemsRequest &req) =0;
};

/**
  Register this CpdListAccessor with Cpd.  The accessor
  will then be called to respond to CCS requests for its path.
  CpdList will eventually delete this object.
*/
void CpdListRegister(CpdListAccessor *acc);

class CpdListAccessor_c : public CpdListAccessor {
  const char *path; //Path to this item
  CpdListLengthFn_c lenFn;
  void *lenParam;
  CpdListItemsFn_c itemsFn;
  void *itemsParam;
public:
  CpdListAccessor_c(const char *path_,
            CpdListLengthFn_c lenFn_,void *lenParam_,
            CpdListItemsFn_c itemsFn_,void *itemsParam_,bool checkBoundary_=true):
	path(path_), lenFn(lenFn_), lenParam(lenParam_), 
	itemsFn(itemsFn_), itemsParam(itemsParam_) {checkBound = checkBoundary_;}
  CpdListAccessor_c(const CpdListAccessor_c &p);//You don't want to copy
  void operator=(const CpdListAccessor_c &p);	// You don't want to copy
  
  virtual const char *getPath(void) const {return path;}
  virtual size_t getLength(void) const {return (*lenFn)(lenParam);}
  virtual void pup(PUP::er &p,CpdListItemsRequest &req) {
    (itemsFn)(itemsParam,(pup_er *)&p,&req);
  }
};

/**
  A typical CpdList accessor: length is stored at some fixed 
   location in memory, path is a constant string, and the 
   pup routine is completely random-access.
*/
class CpdSimpleListAccessor : public CpdListAccessor {
public:
	/// This routine is called to pup each item of the list.
	///  beginItem has already been called before this function.
	typedef void (*pupFn)(PUP::er &p,int itemNo);
private:
	const char *path;
	size_t &length;
	pupFn pfn;
public:
	/**
	  Create a new CpdSimpleListAccessor.
	     \param path_ CpdList path CCS clients should use.
	     \param length_ Reference to number of elements in the list.
	                    This class keeps the reference, so as the list length
			    changes, Cpd always has the latest value.
			    In particular, this means length must not be moved!
	     \param pfn_ Function to pup the items in the list.
	*/
	CpdSimpleListAccessor(const char *path_,size_t &length_,pupFn pfn_)
		:path(path_),length(length_),pfn(pfn_) { }
	virtual ~CpdSimpleListAccessor();
	virtual const char *getPath(void) const;
	virtual size_t getLength(void) const;
	virtual void pup(PUP::er &p,CpdListItemsRequest &req);
};

#endif
