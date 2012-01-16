
#include "converse.h"
#include "debug-conv++.h"
#include "cklists.h"

CpdListAccessor::~CpdListAccessor() { }
CpdSimpleListAccessor::~CpdSimpleListAccessor() { }
const char *CpdSimpleListAccessor::getPath(void) const {return path;}
size_t CpdSimpleListAccessor::getLength(void) const {return length;}
void CpdSimpleListAccessor::pup(PUP::er &p,CpdListItemsRequest &req)
{
        for (int i=req.lo;i<req.hi;i++) {
                beginItem(p,i);
                (*pfn)(p,i);
        }
}

static void CpdListBeginItem_impl(PUP::er &p,int itemNo)
{
        p.syncComment(PUP::sync_item);
}

extern "C" void CpdListBeginItem(pup_er p,int itemNo)
{
  CpdListBeginItem_impl(*(PUP::er *)p,itemNo);
}

void CpdListAccessor::beginItem(PUP::er &p,int itemNo)
{
  CpdListBeginItem_impl(p,itemNo);
}


