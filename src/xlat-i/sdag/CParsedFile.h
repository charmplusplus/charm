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
    TList *entryList;
    TList *nodeList;
    CParsedFile(void) {
      entryList = new TList();
      nodeList = new TList();
    }
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(XStr& classname, XStr& output) {
      output << "#define " << classname << "_SDAG_CODE \n";
      numberNodes();
      labelNodes();
      propogateState();
      generateEntryList();
      generateCode(output);
      generateEntries(output);
      generateInitFunction(output);
      output.line_append('\\');
      output << "\n";
    }
};

#endif
