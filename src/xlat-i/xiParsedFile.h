#include "xi-symbol.h"
#include "CEntry.h"
#include "sdag-globals.h"
#include "EToken.h"
class Entry;


/******************* xiParsedFile ***********************/
class xiParsedFile {
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
    TList<CEntry*> entryList;
    TList<Entry*> nodeList;
    xiParsedFile(void) {}
    ~xiParsedFile(void){}
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

