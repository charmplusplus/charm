/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CEntry_H_
#define _CEntry_H_

#include "xi-util.h"
#include "sdag-globals.h"
#include "CList.h"
#include "CStateVar.h"

class CParseNode;

class CEntry{
  public:
    XStr *entry;
    CParseNode *paramlist;
    int entryNum;
    int needsParamMarshalling;
    int refNumNeeded;
    TList<CStateVar*> *myParameters;
    TList<CParseNode*> whenList;
    CEntry(XStr *e, CParseNode *p, TList<CStateVar*>& list, int pm) : entry(e), paramlist(p), needsParamMarshalling(pm) {
       myParameters = new TList<CStateVar*>();
       CStateVar *sv;
       for(sv=list.begin(); !list.end(); sv=list.next()) {
	  myParameters->append(sv);
       }
       entryNum = numEntries++;
       refNumNeeded =0;
    }

    void print(int indent) {
      Indent(indent);
//      printf("entry %s (%s *)", entry->charstar(), msgType->charstar());
    } 

    void generateCode(XStr& op);
    void generateDeps(XStr& op);
};
#endif
