#ifndef _COverlap_H_
#define _COverlap_H_

#include "xi-util.h"
#include "sdag-globals.h"
#include "CList.h"

class CParseNode;

class COverlap{
  public:
    int overlapNum;
    TList<CParseNode*> whenList;
    COverlap(int on): overlapNum(on) {}

    void generateDeps(XStr& op);
};
#endif
