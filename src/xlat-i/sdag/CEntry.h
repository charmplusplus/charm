#ifndef _CEntry_H_
#define _CEntry_H_

#include "xi-util.h"
#include "sdag-globals.h"

#include <list>

namespace xi {

  class Entry;
  class SdagConstruct;
  class WhenConstruct;
  class ParamList;
  class CStateVar;

  class CEntry{
    std::list<CStateVar*> myParameters;
    // TODO: decide whether this should be private or public
    std::list<Entry*> candidateEntries_;

  public:
    int first_line_, last_line_;
    XStr *entry;
    Entry *decl_entry;			// point to the real Entry of the Chare
    //CParseNode *paramlist;
    ParamList *paramlist;
    int entryNum;
    int needsParamMarshalling;
    int refNumNeeded;
    std::list<WhenConstruct*> whenList;
    CEntry(XStr *e, ParamList *p, const std::list<CStateVar*>& list, int pm, int fl=-1, int ll=-1)
        : entry(e), paramlist(p), needsParamMarshalling(pm), first_line_(fl), last_line_(ll) {
      CStateVar *sv;
      myParameters = list;
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

    std::list<Entry*> getCandidates();
    void addCandidate(Entry *e);
    void check();
  };

}

#endif
