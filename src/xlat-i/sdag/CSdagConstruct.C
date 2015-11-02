#include <string.h>
#include <stdlib.h>
#include "sdag-globals.h"
#include "xi-symbol.h"
#include "xi-Chare.h"
#include "constructs/Constructs.h"
#include "CParsedFile.h"
#include "EToken.h"
#include "CStateVar.h"
#include <list>
using std::list;
#include <algorithm>
using std::for_each;
#include <functional>
using std::mem_fun;

namespace xi {
  SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1) {
    init(t);
    constructs->push_back(construct1);
  }

  SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList) {
    init(t);
    constructs->push_back(construct1);
    constructs->insert(constructs->end(), aList->constructs->begin(), aList->constructs->end());
  }

  SdagConstruct::SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
                               SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el) {
    init(t);
    text = txt;
    con1 = c1; con2 = c2; con3 = c3; con4 = c4;
    if (constructAppend != 0) constructs->push_back(constructAppend);
    elist = el;
  }

  SdagConstruct::SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl) {
    init(t);
    text = new XStr(codeStr);
    param = pl;
  }

  void SdagConstruct::init(EToken& t) {
    con1 = 0; con2 = 0; con3 = 0; con4 = 0;
    traceName = 0;
    elist = 0;
    constructs = new list<SdagConstruct*>();
    type = t;
    label_str = 0;
  }

  SdagConstruct::~SdagConstruct() {
    delete constructs;
    delete text;
  }

  void SdagConstruct::numberNodes(void) {
    if (constructs != 0)
      for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::numberNodes));
  }

  XStr* SdagConstruct::createLabel(const char* str, int nodeNum) {
    char text[128];
    if (nodeNum != -1)
      sprintf(text, "_%s_%d", str, nodeNum);
    else
      sprintf(text, "%s", str);

    return new XStr(text);
  }

  void SdagConstruct::labelNodes() {
    if (label_str != 0)
      label = createLabel(label_str, nodeNum);

    if (constructs != 0)
      for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::labelNodes));
  }

  void EntryList::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    EntryList *el = this;
    while (el != NULL) {
      el->entry->generateEntryList(CEntrylist, thisWhen);
      el = el->next;
    }
  }

  void Entry::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    // case SENTRY:
    bool found = false;
   
    for(list<CEntry *>::iterator entry=CEntrylist.begin(); 
        entry != CEntrylist.end(); ++entry) {
      if(*((*entry)->entry) == (const char *)name) 
        {
          ParamList *epl;
          epl = (*entry)->paramlist;
          ParamList *pl;
          pl = param;
          found = false;
          if (((*entry)->paramlist->isVoid() == 1) && (pl->isVoid() == 1)) {
            found = true;
          }
          while ((pl != NULL) && (epl != NULL))
            {
              bool kindMatches =
                (pl->isArray() && epl->isArray()) ||
                (pl->isBuiltin() && epl->isBuiltin()) ||
                (pl->isReference() && epl->isReference()) ||
                (pl->isMessage() && epl->isMessage()) ||
                (pl->isNamed() && epl->isNamed());
              bool baseNameMatches = (strcmp(pl->getBaseName(), epl->getBaseName()) == 0);
              if (kindMatches && baseNameMatches)
                found = true;

              pl = pl->next;
              epl = epl->next;
            }
          if (((pl == NULL) && (epl != NULL)) ||
              ((pl != NULL) && (epl == NULL)))
            found = false;
          if (found) {
            // check to see if thisWhen is already in entry's whenList
            bool whenFound = false;
            for(list<WhenConstruct*>::iterator it = (*entry)->whenList.begin();
                it != (*entry)->whenList.end(); ++it) {
              if ((*it)->nodeNum == thisWhen->nodeNum)
                whenFound = true;
            }
            if(!whenFound)
              (*entry)->whenList.push_back(thisWhen);
            entryPtr = *entry;
            if(intExpr != 0)
              (*entry)->refNumNeeded = 1; 
          } 
        }
    }
    if(!found) {
      CEntry *newEntry;
      newEntry = new CEntry(new XStr(name), param, estateVars, paramIsMarshalled(), first_line_, last_line_);
      CEntrylist.push_back(newEntry);
      entryPtr = newEntry;
      newEntry->whenList.push_back(thisWhen);
      if(intExpr != 0)
        newEntry->refNumNeeded = 1; 
    }
    //break;
  }

  void SdagConstruct::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    if (SIF == type && con2 != 0)
      con2->generateEntryList(CEntrylist, thisWhen);
    generateChildrenEntryList(CEntrylist, thisWhen);
  }

  void SdagConstruct::generateChildrenEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    if (constructs != 0)
      for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end(); ++it)
        (*it)->generateEntryList(CEntrylist, thisWhen);
  }

  void SdagConstruct::propagateState(int uniqueVarNum) {
    CStateVar *sv; 
    list<EncapState*> encap;

    stateVars = new list<CStateVar*>();
    ParamList *pl = param;
    if (!pl->isVoid()) {
      while (pl != NULL) {
        stateVars->push_back(new CStateVar(pl));
        pl = pl->next;
      }
    
      EncapState* state = new EncapState(this->entry, *stateVars);
      if (!this->entry->paramIsMarshalled() && !this->entry->param->isVoid())
        state->isMessage = true;
      encap.push_back(state);
    }

    encapState = encap;

#if CMK_BIGSIM_CHARM
    // adding _bgParentLog as the last extra parameter for tracing
    stateVarsChildren = new list<CStateVar*>(*stateVars);
    sv = new CStateVar(0, "void *", 0,"_bgParentLog", 0, NULL, 1);
    sv->isBgParentLog = true;
    stateVarsChildren->push_back(sv);

    {
      list<CStateVar*> lst;
      lst.push_back(sv);
      EncapState *state = new EncapState(NULL, lst);
      state->type = new XStr("void");
      state->name = new XStr("_bgParentLog");
      state->isBgParentLog = true;
      encapStateChild.push_back(state);
      encap.push_back(state);
    }
#else
    stateVarsChildren = stateVars; 
#endif

    encapStateChild = encap;

    list<CStateVar*> whensEntryMethodStateVars;
    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it)
      (*it)->propagateState(encap, *stateVarsChildren, whensEntryMethodStateVars, uniqueVarNum);
  }

  void SdagConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist, int uniqueVarNum) {
    CStateVar *sv;
    list<CStateVar*> *whensEntryMethodStateVars = NULL;

    encapState = encap;

    stateVars = new list<CStateVar*>();
    switch(type) {
    case SINT_EXPR:
    case SIDENT:
    case SENTRY:
    case SELIST:
      break;
    default:
      fprintf(stderr, "internal error in sdag translator..\n");
      exit(1);
      break;
    }

    encapStateChild = encap;

    propagateStateToChildren(encap, *stateVarsChildren, wlist, uniqueVarNum);
    delete whensEntryMethodStateVars;
  }


  void SdagConstruct::propagateStateToChildren(list<EncapState*> encap, list<CStateVar*>& stateVarsChildren, list<CStateVar*>& wlist, int uniqueVarNum) {
    if (constructs != 0)
      for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end(); ++it)
        (*it)->propagateState(encap, stateVarsChildren, wlist, uniqueVarNum);
  }

  void SdagConstruct::generateCode(XStr& decls, XStr& defs, Entry *entry) {
    generateChildrenCode(decls, defs, entry);
  }

  void SdagConstruct::generateChildrenCode(XStr& decls, XStr& defs, Entry* entry) {
    if (constructs != 0)
      for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end(); ++it)
        (*it)->generateCode(decls, defs, entry);
  }

  void SdagConstruct::buildTypes(list<EncapState*>& state) {
    for (list<EncapState*>::iterator iter = state.begin(); iter != state.end(); ++iter) {
      EncapState& encap = **iter;
      if (!encap.type) {
        if (encap.entry->entryPtr && encap.entry->entryPtr->decl_entry)
          encap.type = encap.entry->entryPtr->decl_entry->genClosureTypeNameProxyTemp;
        else
          encap.type = encap.entry->genClosureTypeNameProxyTemp;
      }
    }
  }

  int SdagConstruct::unravelClosuresBegin(XStr& defs, bool child) {
    int cur = 0;

    list<EncapState*>& encaps = child ? encapStateChild : encapState;

    // traverse all the state variables bring them into scope
    for (list<EncapState*>::iterator iter = encaps.begin(); iter != encaps.end(); ++iter, ++cur) {
      EncapState& state = **iter;

      indentBy(defs, cur + 1);

      defs << "{\n";

      int i = 0;
      for (list<CStateVar*>::iterator iter2 = state.vars.begin(); iter2 != state.vars.end(); ++iter2, ++i) {
        CStateVar& var = **iter2;

        // if the var is one of the following it a system state var that should
        // not be brought into scope
        if (!var.isCounter && !var.isSpeculator && !var.isBgParentLog) {
          indentBy(defs, cur + 2);

          defs << var.type << (var.arrayLength || var.isMsg ? "*" : "") << "& " << var.name << " = ";
          state.name ? (defs << *state.name) : (defs << "gen" << cur);
          if (!var.isMsg)
            defs << "->" << "getP" << i << "();\n";
          else
            defs << ";\n";
        }
      }
    }

    return cur + 1;
  }

  void SdagConstruct::unravelClosuresEnd(XStr& defs, bool child) {
    list<EncapState*>& encaps = child ? encapStateChild : encapState;

    int cur = encaps.size();

    // traverse all the state variables bring them into scope
    for (list<EncapState*>::iterator iter = encaps.begin(); iter != encaps.end(); ++iter, --cur) {
      EncapState& state = **iter;

      indentBy(defs, cur);

      defs << "}\n";
    }
  }

  void generateVarSignature(XStr& str,
                         const XStr* name, const char* suffix,
                         list<CStateVar*>* params) {
    
  }
  void generateVarSignature(XStr& decls, XStr& defs,
                         const Entry* entry, bool declareStatic, const char* returnType,
                         const XStr* name, bool isEnd,
                         list<CStateVar*>* params) {
    generateVarSignature(decls, defs, entry->getContainer(), declareStatic, returnType,
                      name, isEnd, params);
  }
  void generateVarSignature(XStr& decls, XStr& defs,
                         const Chare* chare, bool declareStatic, const char* returnType,
                         const XStr* name, bool isEnd,
                         list<CStateVar*>* params) {
    decls << "  " << (declareStatic ? "static " : "") << returnType << " ";

    templateGuardBegin(false, defs);
    defs << chare->tspec() << returnType << " " << chare->baseName() << "::";

    XStr op;

    op << name;
    if (isEnd)
      op << "_end";
    op << "(";

    if (params) {
      CStateVar *sv;
      int count = 0;
      for (list<CStateVar*>::iterator iter = params->begin();
           iter != params->end();
           ++iter) {
        CStateVar *sv = *iter;
        if (sv->isVoid != 1) {
          if (count != 0)
            op << ", ";

          // @TODO uncommenting this requires that PUP work on const types
          //if (sv->byConst)
          //op << "const ";
          if (sv->type != 0) 
            op <<sv->type <<" ";
          if (sv->declaredRef)
            op <<" &";
          if (sv->arrayLength != NULL) 
            op <<"* ";
          if (sv->name != 0)
            op <<sv->name;

          count++;
        }
      }
    }

    op << ")";

    decls << op << ";\n";
    defs << op << " {\n";
  }
  void endMethod(XStr& op) {
    op << "}\n";
    templateGuardEnd(op);
    op << "\n\n";
  }

  void generateClosureSignature(XStr& decls, XStr& defs,
                                const Entry* entry, bool declareStatic, const char* returnType,
                                const XStr* name, bool isEnd,
                                list<EncapState*> encap, int numRefs) {
    generateClosureSignature(decls, defs, entry->getContainer(), declareStatic, returnType,
                             name, isEnd, encap, numRefs);
  }
  void generateClosureSignature(XStr& decls, XStr& defs, const Chare* chare,
                                bool declareStatic, const char* returnType,
                                const XStr* name, bool isEnd, list<EncapState*> encap, int numRefs) {
    decls << "  " << (declareStatic ? "static " : "") << returnType << " ";

    templateGuardBegin(false, defs);
    defs << chare->tspec() << returnType << " " << chare->baseName() << "::";

    XStr op;

    op << name;
    if (isEnd) op << "_end";
    op << "(";

    int cur = 0;
    for (list<EncapState*>::iterator iter = encap.begin();
         iter != encap.end(); ++iter, ++cur) {
      EncapState *state = *iter;

      if (state->type) {
        op << *state->type << "* ";
        if (state->name) op << *state->name;
        else op << "gen" << cur;
      } else {
        fprintf(stderr, "type was not propagated to this phase");
        exit(120);
      }

      if (cur != encap.size() - 1) op << ", ";
    }

    for (int i = 0; i < numRefs; i++) op << ((cur+i) > 0 ? ", " : "") << "int refnum_" << i;

    op << ")";

    decls << op << ";\n";
    defs << op << " {\n";
  }

  void SdagConstruct::generateCall(XStr& op, list<EncapState*>& scope,
                                      list<EncapState*>& next, const XStr* name,
                                      const char* nameSuffix) {
    op << name << (nameSuffix ? nameSuffix : "") << "(";

    int cur = 0;
    for (list<EncapState*>::iterator iter = next.begin(); iter != next.end(); ++iter, ++cur) {
      EncapState *state = *iter;

      if (state->type) {
        if (cur >= scope.size()) {
          int offset = cur - scope.size();
          if (!state->isMessage)
            op << "static_cast<" << *state->type << "*>(buf" << offset << "->cl)";
          else
            op << "static_cast<" << *state->type << "*>(static_cast<SDAG::MsgClosure*>(buf" << offset << "->cl)->msg)";
        } else {
          if (state->name) op << *state->name; else op << "gen" << cur;
        }
      } else {
        fprintf(stderr, "type was not propagated to this phase");
        exit(120);
      }

      if (cur != next.size() - 1) op << ", ";
    }

    op << ");\n";
  }

  // boe = 1, if the next call is to begin construct
  // boe = 0, if the next call is to end a contruct
  void SdagConstruct::setNext(SdagConstruct *n, int boe) {
    switch(type) {
    case SSLIST:
      next = n;
      nextBeginOrEnd = boe;
      {
        if (constructs->empty())
          return;

        list<SdagConstruct*>::iterator it = constructs->begin();
        SdagConstruct *cn = *it;
        ++it;

        for(; it != constructs->end(); ++it) {
          cn->setNext(*it, 1);
          cn = *it;
        }
        cn->setNext(this, 0);
      }
      return;
    case SCASELIST:
      next = n;
      nextBeginOrEnd = boe;
      {
        for(list<SdagConstruct*>::iterator it = constructs->begin();
            it != constructs->end();
            ++it) {
          (*it)->setNext(this, 0);
        }
      }
      return;
    case SCASE:
    case SSDAGENTRY:
    case SOVERLAP:
    case SOLIST:
    case SFORALL:
    case SWHEN:
    case SFOR:
    case SWHILE:
    case SATOMIC:
    case SELSE:
      next = n;
      nextBeginOrEnd = boe;
      n = this; boe = 0; break;
    case SIF:
      next = n;
      nextBeginOrEnd = boe;
      if(con2 != 0)
        con2->setNext(n, boe);
      n = this;
      boe = 0;
      break;
    default:
      break;
    }
    SdagConstruct *cn;
    if (constructs != 0) {
      for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
           ++it) {
        (*it)->setNext(n, boe);
      }
    }
  }

  // for trace
  void SdagConstruct::generateTrace() {
    for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::generateTrace));
    if (con1) con1->generateTrace();
    if (con2) con2->generateTrace();
    if (con3) con3->generateTrace();
  }

  void SdagConstruct::generateTraceBeginCall(XStr& op, int indent) {
    if (traceName) {
      indentBy(op, indent);
      op << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (" << "_sdag_idx_" << traceName << "()), CkMyPe(), 0, NULL, NULL); \n";
    }
  }

  void SdagConstruct::generateDummyBeginExecute(XStr& op, int indent) {
    indentBy(op, indent);
    op << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, NULL, NULL); \n";
  }

  void SdagConstruct::generateTraceEndCall(XStr& op, int indent) {
    indentBy(op, indent);
    op << "_TRACE_END_EXECUTE(); \n";
  }

  void SdagConstruct::generateBeginExec(XStr& op, const char *name) {
    op << "     " << "_TRACE_BG_BEGIN_EXECUTE_NOMSG(\""<<name<<"\", &_bgParentLog,1);\n";
  }

  void SdagConstruct::generateEndExec(XStr& op){
    op << "     " << "_TRACE_BG_END_EXECUTE(0);\n";
  }

  void SdagConstruct::generateBeginTime(XStr& op) {
    //Record begin time for tracing
    op << "  double __begintime = CkVTimer(); \n";
  }

  void SdagConstruct::generateTlineEndCall(XStr& op) {
    //Trace this event
    op <<"    _TRACE_BG_TLINE_END(&_bgParentLog);\n";
  }

  void SdagConstruct::generateEndSeq(XStr& op) {
    op <<  "    void* _bgParentLog = NULL;\n";
    op <<  "    CkElapse(0.01e-6);\n";
    //op<<  "    BgElapse(1e-6);\n";
    generateTlineEndCall(op);
    generateTraceEndCall(op, 1);
    generateEndExec(op);
  }

  void SdagConstruct::generateEventBracket(XStr& op, int eventType) {
    (void) eventType;
    //Trace this event
    op << "     _TRACE_BG_USER_EVENT_BRACKET(\"" << nameStr
       << "\", __begintime, CkVTimer(), &_bgParentLog); \n";
  }

  void SdagConstruct::generateListEventBracket(XStr& op, int eventType) {
    (void) eventType;
    op << "     _TRACE_BGLIST_USER_EVENT_BRACKET(\"" << nameStr
       << "\", __begintime,CkVTimer(), &_bgParentLog, " << label
       << "_bgLogList);\n";
  }

  void SdagConstruct::generateRegisterEp(XStr& defs) {
    if (traceName)
      defs << "  (void)_sdag_idx_" << traceName << "();\n";

    for (list<SdagConstruct*>::iterator iter = constructs->begin(); iter != constructs->end(); ++iter)
      (*iter)->generateRegisterEp(defs);
    if (con1) con1->generateRegisterEp(defs);
    if (con2) con2->generateRegisterEp(defs);
    if (con3) con3->generateRegisterEp(defs);
  }

  void SdagConstruct::generateTraceEp(XStr& decls, XStr& defs, Chare* chare) {
    if (traceName) {
      XStr regName, idxName;

      idxName << "_sdag_idx_" << traceName;
      regName << "_sdag_reg_" << traceName;
      generateVarSignature(decls, defs, chare, true, "int", &idxName, false, NULL);
      defs << "  static int epidx = " << regName << "();\n"
           << "  return epidx;\n";
      endMethod(defs);

      generateVarSignature(decls, defs, chare, true, "int", &regName, false, NULL);
      defs << "  return CkRegisterEp(\""
           << traceName << "\", NULL, 0, " << chare->indexName() << "::__idx, 0"
           << ");\n";
      endMethod(defs);
    }

    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it) {
      (*it)->generateTraceEp(decls, defs, chare);
    }
    if (con1) con1->generateTraceEp(decls, defs, chare);
    if (con2) con2->generateTraceEp(decls, defs, chare);
    if (con3) con3->generateTraceEp(decls, defs, chare);
  }
}   // namespace xi
