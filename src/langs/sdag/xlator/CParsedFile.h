/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CParsedFile_H_
#define _CParsedFile_H_

#include "xi-util.h"
#include "CList.h"
#include "CEntry.h"
#include "CParseNode.h"
#include "sdag-globals.h"
#include <stdio.h>

class CParsedFile {
  private:
    void numberNodes(void);
    void labelNodes(void);
    void propogateState(void);
    void generateEntryList(void);
    void generateCode(XStr& output);
    void generateEntries(XStr& output);
    void generateInitFunction(XStr& output);
  public:
    XStr *className;
    TList *entryList;
    TList *nodeList;
    CParsedFile(void): className(0) {
      entryList = new TList();
      nodeList = new TList();
    }
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(XStr& output) {
      numberNodes();
      labelNodes();
      propogateState();
      generateEntryList();
      generateCode(output);
      generateEntries(output);
      generateInitFunction(output);
    }
};

#endif
