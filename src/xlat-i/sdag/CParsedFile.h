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
#include "COverlap.h"
#include "CEntry.h"
#include "CParseNode.h"
#include "sdag-globals.h"
#include <stdio.h>

class CParsedFile {
  private:
    void numberNodes(void);
    void labelNodes(void);
    void propagateState(void);
    void generateEntryList(void);
    void generateCode(XStr& output);
    void generateEntries(XStr& output);
    void generateInitFunction(XStr& output);
    void generatePupFunction(XStr& output);
  public:
    TList<COverlap*> overlapList;
    TList<CEntry*> entryList;
    TList<CParseNode*> nodeList;
    CParsedFile(void) {}
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(XStr& classname, XStr& output) {
      output << "#define " << classname << "_SDAG_CODE \n";
      numberNodes();
      labelNodes();
      propagateState();
      generateEntryList();
      generateCode(output);
      generateEntries(output);
      generateInitFunction(output);
      generatePupFunction(output);
      output.line_append('\\');
      output << "\n";
    }
};

#endif
