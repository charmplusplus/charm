/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CParseNode_H_
#define _CParseNode_H_

#include "EToken.h"
#include "xi-util.h"
#include "CLexer.h"
#include "sdag-globals.h"
#include "CList.h"
#include "CStateVar.h"
#include "COverlap.h"
#include "CEntry.h"

class CParser;

class CParseNode {
  public:
    void print(int indent);
    int nodeNum;
    XStr *label;
    XStr *counter;
    EToken type;
    XStr *text;
    XStr *vartype;
    int numPtrs;
    int isVoid; 
    int needsParamMarshalling;
    int isOverlaped;
    CParseNode *con1, *con2, *con3, *con4;
    TList<CParseNode*> *constructs;
    TList<CStateVar*> estateVars;
    TList<CStateVar*> *stateVars;
    TList<CStateVar*> *stateVarsChildren;
    CParseNode *next;
    int nextBeginOrEnd;
    CEntry *entryPtr;
    CParseNode(EToken t, CLexer *cLexer, CParser *cParser, int overlaps);
    CParseNode(EToken t, CLexer *cLexer, CParser *cParser, CToken *tokA, CToken *tokB, int pointers);
    CParseNode(EToken t, XStr *txt) : type(t), text(txt), con1(0), con2(0),
                                         con3(0), con4(0), constructs(0) {}
    void numberNodes(void);
    void labelNodes(void);
    void generateEntryList(TList<CEntry*>&, TList<COverlap*>&, CParseNode *);
    void propagateState(int);
    void generateCode(XStr& output);
    void setNext(CParseNode *, int);
  private:
    void propagateState(TList<CStateVar*>&, TList<CStateVar*>&, int);
    void generateWhen(XStr& op);
    void generateOverlap(XStr& op);
    void generateWhile(XStr& op);
    void generateFor(XStr& op);
    void generateIf(XStr& op);
    void generateElse(XStr& op);
    void generateForall(XStr& op);
    void generateOlist(XStr& op);
    void generateSdagEntry(XStr& op);
    void generateSlist(XStr& op);
    void generateAtomic(XStr& op);
    void generateForward(XStr& op);
    void generatePrototype(XStr& op, TList<CStateVar*>&);
    void generateCall(XStr& op, TList<CStateVar*>&);
};
#endif
