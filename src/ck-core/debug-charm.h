/*
 Interface to Charm++ portion of parallel debugger.
 Orion Sky Lawlor, olawlor@acm.org, 7/30/2001
 */
#ifndef __CMK_DEBUG_CHARM_H
#define __CMK_DEBUG_CHARM_H

#ifndef __cplusplus
#  error "debug-charm.h is for C++; use debug-conv.h for C programs"
#endif

#include "pup.h"

/* CpdList functions for C++ */
class CpdListAccessor { /* Abstract superclass */
protected:
  void beginItem(PUP::er &p,int itemNo);
public:
  virtual ~CpdListAccessor(); 
  virtual const char *getPath(void) const =0;
  virtual int getLength(void) const =0;
  virtual void pup(PUP::er &p,CpdListItemsRequest &req) =0;
};
void CpdListRegister(CpdListAccessor *acc);

/*A typical accessor: length is stored at some fixed location,
path is a constant, pup is random-access.*/
class CpdSimpleListAccessor : public CpdListAccessor {
public:
	typedef void (*pupFn)(PUP::er &p,int itemNo);
private:
	const char *path;
	int &length;
	pupFn pfn;
public:
	CpdSimpleListAccessor(const char *path_,int &length_,pupFn pfn_)
		:path(path_),length(length_),pfn(pfn_) { }
	virtual ~CpdSimpleListAccessor();
	virtual const char *getPath(void) const;
	virtual int getLength(void) const;
	virtual void pup(PUP::er &p,CpdListItemsRequest &req);
};

#endif
