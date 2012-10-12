#ifndef CK_CPARSEDFILE_H
#define CK_CPARSEDFILE_H

#include "xi-symbol.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "EToken.h"
#include <list>

namespace xi {

class Chare;
class Entry;

class CParsedFile {
  private:
    void mapCEntry();			// search and check if all functions in when() are defined.
    void generateConnectEntryList(void);
    void generateEntryList(void);       // collect and setup CEntry list for When and If
    void generateCode(XStr& decls, XStr& defs);
    void generateEntries(XStr& decls, XStr& defs);
    void generateConnectEntries(XStr& output);
    void generateInitFunction(XStr& decls, XStr& defs);
    void generatePupFunction(XStr& decls, XStr& defs);
    void generateTraceEp(XStr& decls, XStr& defs);
    void generateRegisterEp(XStr& decls, XStr& defs);
    void generateDependencyMergePoints(XStr& output);
    std::list<Entry*> nodeList;
    std::list<CEntry*> entryList;
    std::list<SdagConstruct *> connectEntryList;
    Chare *container;

  public:
    static XStr *className;
    CParsedFile(Chare *c): container(c) {}
    ~CParsedFile(void){}
    void print(int indent);
    void addNode(Entry *e) { nodeList.push_back(e); }
    void doProcess(XStr& classname, XStr& decls, XStr& defs);
};

}

#endif
