
#include "converse.h"
#include "debug-conv++.h"
#include "cklists.h"

CpdListAccessor::~CpdListAccessor() { }

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


