#include "xi-symbol.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "EToken.h"

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
    void generateCode(XStr& output);
    void generateEntries(XStr& output);
    void generateConnectEntries(XStr& output);
    void generateInitFunction(XStr& output);
    void generatePupFunction(XStr& output);
    void generateRegisterEp(XStr& output);
    void generateTraceEpDecl(XStr& output);
    void generateTraceEpDef(XStr& output);
    void generateDependencyMergePoints(XStr& output);
    void generateTrace();
  public:
    Chare *container;
    static XStr *className;
    TList<CEntry*> entryList;
    TList<SdagConstruct *> connectEntryList;
    TList<Entry*> nodeList;
    CParsedFile(Chare *c): container(c) {}
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(XStr& classname, XStr& output) {
      className = &classname;
      output << "#define " << classname << "_SDAG_CODE \n";
      

      numberNodes();
      labelNodes();
      propagateState();
      generateConnectEntryList();
      generateTrace();				// for tracing Gengbin
      generateEntryList();
      mapCEntry();
      generateCode(output);
      generateEntries(output);
      generateInitFunction(output);
      generatePupFunction(output);
      generateRegisterEp(output);		// for tracing Gengbin
      generateTraceEpDecl(output);		// for tracing Gengbin

      generateDependencyMergePoints(output); // for Isaac's Critical Path Detection

      output.line_append_padding('\\');
      output << "\n";
      output << "#define " << classname << "_SDAG_CODE_DEF \\\n";
      generateTraceEpDef(output);
      output << "\n";
    }

};

