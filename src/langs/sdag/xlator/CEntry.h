/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CEntry_H_
#define _CEntry_H_

#include "CString.h"
#include "sdag-globals.h"
#include "CList.h"

class CParseNode;

class CEntry{
  public:
    CString *entry;
    CString *msgType;
    int entryNum;
    int refNumNeeded;
    TList *whenList;
    CEntry(CString *e, CString *m) : entry(e), msgType(m) {
      entryNum = numEntries++;
      whenList = new TList();
      refNumNeeded=0;
    }
    void print(int indent) {
      Indent(indent);
      printf("entry %s (%s *)", entry->charstar(), msgType->charstar());
    }
    void generateCode(CString *);
    void generateDeps(void);
    
};
#endif
