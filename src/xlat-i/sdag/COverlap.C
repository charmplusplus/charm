#include "COverlap.h"
#include "CParseNode.h"


void COverlap::generateDeps(XStr& op) {
   CParseNode *cn;
   for(cn=whenList.begin(); !whenList.end(); cn=whenList.next()) {
      op<<"    __cOverDep->addOverlapDepends("<<cn->nodeNum<<","<<overlapNum<<");\n";
   }
}
