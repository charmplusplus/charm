#ifndef _CEntry_H_
#define _CEntry_H_

#include "xi-util.h"
#include "sdag-globals.h"
#include "CList.h"

#include <list>

namespace xi {

class Entry;
class SdagConstruct;
class ParamList;
class CStateVar;

class CEntry{
  std::list<CStateVar*> myParameters;

public:
    XStr *entry;
    Entry *decl_entry;			// point to the real Entry of the Chare
    //CParseNode *paramlist;
    ParamList *paramlist;
    int entryNum;
    int needsParamMarshalling;
    int refNumNeeded;
    std::list<SdagConstruct*> whenList;
    CEntry(XStr *e, ParamList *p, TList<CStateVar*>& list, int pm) : entry(e), paramlist(p), needsParamMarshalling(pm) {
       CStateVar *sv;
       for(sv=list.begin(); !list.end(); sv=list.next()) {
	  myParameters.push_back(sv);
       }
       entryNum = numEntries++;
       refNumNeeded =0;
       decl_entry = NULL;
    }

    void print(int indent) {
      Indent(indent);
//      printf("entry %s (%s *)", entry->charstar(), msgType->charstar());
    } 

    void generateCode(XStr& decls, XStr& defs);
    void generateDeps(XStr& op);
};

}

#endif
