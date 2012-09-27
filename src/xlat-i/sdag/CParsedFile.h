#include "xi-symbol.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "EToken.h"
#include <list>
namespace xi {

class Chare;
class Entry;


/******************* CParsedFile ***********************/
class CParsedFile {
  private:
    void numberNodes(void);
    void labelNodes(void);
    void mapCEntry();			// search and check if all functions in when() are defined.
    void propagateState(void);
    void generateConnectEntryList(void);
    void generateEntryList(void);       // collect and setup CEntry list for When and If
    void generateCode(XStr& decls, XStr& defs);
    void generateEntries(XStr& decls, XStr& defs);
    void generateConnectEntries(XStr& output);
    void generateInitFunction(XStr& decls, XStr& defs);
    void generatePupFunction(XStr& output);
    void generateTraceEp(XStr& decls, XStr& defs);
    void generateRegisterEp(XStr& decls, XStr& defs);
    void generateDependencyMergePoints(XStr& output);
    void generateTrace();
    std::list<Entry*> nodeList;
    TList<CEntry*> entryList;
    Chare *container;

  public:
    static XStr *className;
    TList<SdagConstruct *> connectEntryList;
    CParsedFile(Chare *c): container(c) {}
    ~CParsedFile(void){}
    void print(int indent);
    void addNode(Entry *e) { nodeList.push_back(e); }
    void doProcess(XStr& classname, XStr& decls, XStr& defs) {
      className = &classname;
      decls << "#define " << classname << "_SDAG_CODE \n";

      numberNodes();
      labelNodes();
      propagateState();
      generateConnectEntryList();
      generateTrace();
      generateEntryList();
      mapCEntry();
      generateCode(decls, defs);
      generateEntries(decls, defs);
      generateInitFunction(decls, defs);
      generatePupFunction(decls);
      generateRegisterEp(decls, defs);
      generateTraceEp(decls, defs);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
      generateDependencyMergePoints(decls); // for Isaac's Critical Path Detection
#endif

      decls.line_append_padding('\\');
      decls << "\n";
    }

};

}
