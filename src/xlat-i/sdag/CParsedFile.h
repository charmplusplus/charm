#include "xi-symbol.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "EToken.h"
class Entry;


/******************* CParsedFile ***********************/
class CParsedFile {
  private:
    void numberNodes(void);
    void labelNodes(void);
    void propagateState(void);
    void generateConnectEntryList(void);
    void generateEntryList(void);
    void generateCode(XStr& output);
    void generateEntries(XStr& output);
    void generateConnectEntries(XStr& output);
    void generateInitFunction(XStr& output);
    void generatePupFunction(XStr& output);
  public:
    TList<CEntry*> entryList;
    TList<SdagConstruct *> connectEntryList;
    TList<Entry*> nodeList;
   CParsedFile(void) {}
    ~CParsedFile(void){}
    void print(int indent);
    void doProcess(XStr& classname, XStr& output) {
      output << "#define " << classname << "_SDAG_CODE \n";
      numberNodes();
      labelNodes();
      propagateState();
      generateConnectEntryList();
      generateEntryList();
      generateCode(output);
      generateEntries(output);
      generateInitFunction(output);
      generatePupFunction(output);
      output.line_append('\\');
      output << "\n";
    }

};

