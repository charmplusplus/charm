#ifndef _CParseNode_H_
#define _CParseNode_H_

#include "EToken.h"
#include "CString.h"
#include "CLexer.h"
#include "sdag-globals.h"
#include "CList.h"
#include "CStateVar.h"
#include "CEntry.h"

class CParser;

class CParseNode {
  public:
    CString *className;
    void print(int indent);
    int nodeNum;
    CString *label;
    CString *counter;
    EToken type;
    CString *text;
    CParseNode *con1, *con2, *con3, *con4;
    TList *constructs;
    TList *stateVars;
    TList *stateVarsChildren;
    CParseNode *next;
    int nextBeginOrEnd;
    CEntry *entryPtr;
    CParseNode(EToken t, CLexer *cLexer, CParser *cParser);
    CParseNode(EToken t, CString *txt) : type(t), text(txt), con1(0), con2(0),
                                         con3(0), con4(0), constructs(0) {}
    void numberNodes(void);
    void labelNodes(CString *);
    void generateEntryList(TList *, CParseNode *);
    void propogateState(TList *);
    void generateCode(void);
    void setNext(CParseNode *, int);
  private:
    void generateWhen(void);
    void generateOverlap(void);
    void generateWhile(void);
    void generateFor(void);
    void generateIf(void);
    void generateElse(void);
    void generateForall(void);
    void generateOlist(void);
    void generateSdagEntry(void);
    void generateSlist(void);
    void generateAtomic(void);
    void generatePrototype(FILE *, TList *);
    void generateCall(FILE *f, TList *);
};
#endif
