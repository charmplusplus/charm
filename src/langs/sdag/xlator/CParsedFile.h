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
    void generateCode(void);
    void generateEntries(void);
    void generateInitFunction(void);
  public:
    XStr *className;
    XStr *sourceFile;
    TList *entryList;
    TList *nodeList;
    CParsedFile(char *sFile): className(0) {
      sourceFile = new XStr(sFile);
      entryList = new TList();
      nodeList = new TList();
    }
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(void) {
      numberNodes();
      labelNodes();
      propogateState();
      generateEntryList();
      XStr *fhname = new XStr(sourceFile->charstar());
      fhname->append(".h");
      fh = fopen(fhname->charstar(), "w");
      generateCode();
      generateEntries();
      generateInitFunction();
      fclose(fh);
    }
};

#endif
