#include <string.h>
#include <stdlib.h>
#include "sdag-globals.h"
#include "xi-symbol.h"
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
    con1 = 0;  con2 = 0; con3 = 0; con4 = 0;
    elist = 0;
    type = t;
    traceName=NULL;
    constructs = new list<SdagConstruct*>();
    constructs->push_back(construct1);
  }

  SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList) {
    con1=0; con2=0; con3=0; con4=0;
    type = t;
    elist = 0;
    traceName=NULL;
    constructs = new list<SdagConstruct*>();
    constructs->push_back(construct1);
    constructs->insert(constructs->end(), aList->constructs->begin(), aList->constructs->end());
  }

  SdagConstruct::SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
                               SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el) {
    text = txt;
    type = t;
    traceName=NULL;
    con1 = c1; con2 = c2; con3 = c3; con4 = c4;
    constructs = new list<SdagConstruct*>();
    if (constructAppend != 0) {
      constructs->push_back(constructAppend);
    }
    elist = el;
  }

  SdagConstruct::SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl) {
    type = t;
    traceName=NULL;
    text = new XStr(codeStr);
    con1 = 0; con2 = 0; con3 = 0; con4 =0;
    constructs = new list<SdagConstruct*>();
    param = pl;
    elist = 0;
  }

  void SdagConstruct::numberNodes(void) {
    switch(type) {
    case SSDAGENTRY: nodeNum = numSdagEntries++; break;
    case SOVERLAP: nodeNum = numOverlaps++; break;
    case SWHEN: nodeNum = numWhens++; break;
    case SFOR: nodeNum = numFors++; break;
    case SWHILE: nodeNum = numWhiles++; break;
    case SIF: nodeNum = numIfs++; if(con2!=0) con2->numberNodes(); break;
    case SELSE: nodeNum = numElses++; break;
    case SFORALL: nodeNum = numForalls++; break;
    case SSLIST: nodeNum = numSlists++; break;
    case SOLIST: nodeNum = numOlists++; break;
    case SATOMIC: nodeNum = numAtomics++; break;
    case SCASE: nodeNum = numCases++; break;
    case SCASELIST: nodeNum = numCaseLists++; break;
    case SINT_EXPR:
    case SIDENT: 
    default:
      break;
    }
    SdagConstruct *cn;
    if (constructs != 0)
      for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::numberNodes));
  }

  XStr* SdagConstruct::createLabel(const char* str, int nodeNum) {
    char text[128];
    if (nodeNum != -1)
      sprintf(text, "%s%d", str, nodeNum);
    else
      sprintf(text, "%s", str);

    return new XStr(text);
  }

  void SdagConstruct::labelNodes() {
    switch(type) {
    case SSDAGENTRY: label = createLabel(con1->text->charstar(), -1); break;
    case SOVERLAP: label = createLabel("_overlap_", nodeNum); break;
    case SWHEN: label = createLabel("_when_", nodeNum);
      for (EntryList *el = elist; el != NULL; el = el->next)
        el->entry->label = new XStr(el->entry->name);
      break;
    case SFOR: label = createLabel("_for_", nodeNum); break;
    case SWHILE: label = createLabel("_while_", nodeNum); break;
    case SIF: label = createLabel("_if_", nodeNum);
      if (con2 != 0) con2->labelNodes();
      break;
    case SELSE: label = createLabel("_else_", nodeNum); break;
    case SFORALL: label = createLabel("_forall_", nodeNum); break;
    case SSLIST: label = createLabel("_slist_", nodeNum); break;
    case SOLIST: label = createLabel("_olist_", nodeNum); break;
    case SATOMIC: label = createLabel("_atomic_", nodeNum); break;
    case SCASE: label = createLabel("_case_", nodeNum); break;
    case SCASELIST: label = createLabel("_caselist_", nodeNum); break;
    case SINT_EXPR: case SIDENT: default: break;
    }
    SdagConstruct *cn;
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
      newEntry = new CEntry(new XStr(name), param, estateVars, paramIsMarshalled() );
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

  void WhenConstruct::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    elist->generateEntryList(CEntrylist, this);  /* con1 is the WHEN's ELIST */
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
    if (pl->isVoid() == 1) {
      sv = new CStateVar(1, NULL, 0, NULL, 0, NULL, 0);
      stateVars->push_back(sv);
      std::list<CStateVar*> lst;
      encap.push_back(new EncapState(this->entry, lst));
    }
    else {
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
    case SFORALL:
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      stateVarsChildren = new list<CStateVar*>(plist);
      sv = new CStateVar(0,"int", 0, con1->text->charstar(), 0,NULL, 0);
      stateVarsChildren->push_back(sv);

      {
        list<CStateVar*> lst;
        lst.push_back(sv);
        EncapState *state = new EncapState(NULL, lst);
        state->isForall = true;
        state->type = new XStr("SDAG::ForallClosure");
        XStr* name = new XStr();
        *name << con1->text << "_cl";
        state->name = name;
        encap.push_back(state);
      }

      {
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "SDAG::CCounter *", 0, txt, 0, NULL, 1);
        sv->isCounter = true;
        stateVarsChildren->push_back(sv);

        list<CStateVar*> lst;
        lst.push_back(sv);
        EncapState *state = new EncapState(NULL, lst);
        state->type = new XStr("SDAG::CCounter");
        state->name = new XStr(txt);
        encap.push_back(state);
      }
      break;
    case SIF:
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      stateVarsChildren = stateVars;
      if(con2 != 0) con2->propagateState(encap, plist, wlist, uniqueVarNum);
      break;
    case SCASELIST:
      stateVarsChildren = new list<CStateVar*>(plist);
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      {
        char txt[128];
        sprintf(txt, "_cs%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "SDAG::CSpeculator *", 0, txt, 0, NULL, 1);
        sv->isSpeculator = true;
        stateVarsChildren->push_back(sv);

        for (std::list<SdagConstruct *>::iterator iter = constructs->begin();
             iter != constructs->end();
             ++iter) {
          dynamic_cast<WhenConstruct*>(*iter)->speculativeState = sv;
        }
        list<CStateVar*> lst;
        lst.push_back(sv);
        EncapState *state = new EncapState(NULL, lst);
        state->name = new XStr(txt);
        state->type = new XStr("SDAG::CSpeculator");
        encap.push_back(state);
      }
      
      break;
    case SOLIST:
      stateVarsChildren = new list<CStateVar*>(plist);
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      {
        char txt[128];
        sprintf(txt, "_co%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "SDAG::CCounter *", 0, txt, 0, NULL, 1);
        sv->isCounter = true;
        stateVarsChildren->push_back(sv);

        list<CStateVar*> lst;
        lst.push_back(sv);
        EncapState *state = new EncapState(NULL, lst);
        state->type = new XStr("SDAG::CCounter");
        state->name = new XStr(txt);
        encap.push_back(state);
      }
      break;
    case SFOR:
    case SWHILE:
    case SELSE:
    case SSLIST:
    case SOVERLAP:
    case SCASE:
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      stateVarsChildren = stateVars;
      break;
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

  void WhenConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist,  int uniqueVarNum) {
    CStateVar *sv;
    list<CStateVar*> whensEntryMethodStateVars;
    list<CStateVar*> whenCurEntry;
    stateVars = new list<CStateVar*>();
    stateVarsChildren = new list<CStateVar*>();

    for (list<CStateVar*>::iterator iter = plist.begin(); iter != plist.end(); ++iter) {
      sv = *iter;
      stateVars->push_back(sv);
      stateVarsChildren->push_back(sv);
    }

    encapState = encap;

    EntryList *el;
    el = elist;
    ParamList *pl;
    while (el != NULL) {
      pl = el->entry->param;
      if (!pl->isVoid()) {
        while(pl != NULL) {
          sv = new CStateVar(pl);
          stateVarsChildren->push_back(sv);
          whensEntryMethodStateVars.push_back(sv);
          whenCurEntry.push_back(sv);
          el->entry->addEStateVar(sv);

          pl = pl->next;
        }
      }

      EncapState* state = new EncapState(el->entry, whenCurEntry);
      if (!el->entry->paramIsMarshalled() && !el->entry->param->isVoid())
        state->isMessage = true;
      encap.push_back(state);
      whenCurEntry.clear();
      el = el->next;
    }

    encapStateChild = encap;

    propagateStateToChildren(encap, *stateVarsChildren, whensEntryMethodStateVars, uniqueVarNum);
  }


  void AtomicConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist, int uniqueVarNum) {
    stateVars = new list<CStateVar*>();
    stateVars->insert(stateVars->end(), plist.begin(), plist.end());
    stateVarsChildren = stateVars;

    encapState = encap;
    encapStateChild = encap;
  }

  void SdagConstruct::propagateStateToChildren(list<EncapState*> encap, list<CStateVar*>& stateVarsChildren, list<CStateVar*>& wlist, int uniqueVarNum) {
    if (constructs != 0)
      for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end(); ++it)
        (*it)->propagateState(encap, stateVarsChildren, wlist, uniqueVarNum);
  }

  void SdagConstruct::generateCode(XStr& decls, XStr& defs, Entry *entry) {
    switch(type) {
    case SSDAGENTRY: generateSdagEntry(decls, defs, entry); break;
    case SSLIST: generateSlist(decls, defs, entry); break;
    case SOLIST: generateOlist(decls, defs, entry); break;
    case SFORALL: generateForall(decls, defs, entry); break;
    case SIF: generateIf(decls, defs, entry);
      if(con2 != 0) con2->generateCode(decls, defs, entry);
      break;
    case SELSE: generateElse(decls, defs, entry); break;
    case SWHILE: generateWhile(decls, defs, entry); break;
    case SFOR: generateFor(decls, defs, entry); break;
    case SCASE: case SOVERLAP: generateOverlap(decls, defs, entry); break;
    case SCASELIST: generateCaseList(decls, defs, entry); break;
    default: break;
    }
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

  void SdagConstruct::generateWhenCodeNew(XStr& op) {
    buildTypes(encapState);
    buildTypes(encapStateChild);

    // generate call this when
    op << "      " << this->label << "(";
    int cur = 0;
    for (list<EncapState*>::iterator iter = encapState.begin();
         iter != encapState.end(); ++iter, ++cur) {
      EncapState& state = **iter;
      if (!state.isMessage)
        op << "\n        static_cast<" << *state.type << "*>(c->closure[" << cur << "])";
      else
        op << "\n        static_cast<" << *state.type << "*>(static_cast<SDAG::MsgClosure*>(c->closure[" << cur << "])->msg)";
      if (cur != encapState.size() - 1) op << ", ";
    }  
    op << "\n      );\n";
  }

  void WhenConstruct::generateEntryName(XStr& defs, Entry* e, int curEntry) {
    if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() == 1))
      defs << e->getEntryName() << "_" << curEntry;
    else {
      for (list<CStateVar*>::iterator it = e->stateVars.begin(); it != e->stateVars.end(); ++it) {
        CStateVar* sv = *it;
        defs << sv->name;
      }
    }
    defs << "_buf";
  }

  void WhenConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    buildTypes(encapState);
    buildTypes(encapStateChild);

    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    generateSignatureNew(decls, defs, entry, false, "SDAG::Continuation*", label, false, encapState);

    int entryLen = 0;

    // count the number of entries this when contains (for logical ands)
    {
      int cur = 0;
      for (EntryList *el = elist; el != NULL; el = el->next, cur++) entryLen++;
    }

    // if we have a reference number in the closures, we need to unravel the state
    int cur = 0;
    bool hasRef = false;;
    for (EntryList *el = elist; el != NULL; el = el->next, cur++)
      if (el->entry->intExpr) {
        defs << "  int refnum_" << cur << ";\n";
        hasRef = true;
      }
    int indent = 2;

    // unravel the closures so the potential refnum expressions can be resolved
    if (hasRef) {
      indent = unravelClosuresBegin(defs);
      indentBy(defs, indent);
      // create a new scope for unraveling the closures
      defs << "{\n";
    }
    cur = 0;
    // generate each refnum variable we need that can access the internal closure state
    for (EntryList *el = elist; el != NULL; el = el->next, cur++)
      if (el->entry->intExpr) {
        indentBy(defs, indent + 1);
        defs << "refnum_" << cur << " = " << (el->entry->intExpr ? el->entry->intExpr : "0") << ";\n";
      }

    if (hasRef) {
      indentBy(defs, indent);
      defs << "}\n";
      unravelClosuresEnd(defs);
    }

    if (entryLen > 1) defs << "  std::set<SDAG::Buffer*> ignore;\n";

    XStr haveAllBuffersCond;
    XStr removeMessagesIfFound, deleteMessagesIfFound;
    XStr continutationSpec;

    {
      int cur = 0;
      for (EntryList *el = elist; el != NULL; el = el->next, cur++) {
        Entry* e = el->entry;
        XStr bufName("buf");
        bufName << cur;
        XStr refName;
        refName << "refnum_" << cur;
        defs << "  SDAG::Buffer* " << bufName << " = __dep->tryFindMessage("
             << e->entryPtr->entryNum // entry number
             << ", " << (e->intExpr ? "true" : "false") // has a ref number?
             << ", " << (e->intExpr ? refName.get_string_const() : "0")  // the ref number
             << ", " << (cur != 0 ? "true" : "false")   // has a ignore set?
             << (entryLen > 1 ? ", ignore" : "") // the ignore set
             << ");\n";
        haveAllBuffersCond << bufName;
        removeMessagesIfFound << "    __dep->removeMessage(" << bufName << ");\n";
        deleteMessagesIfFound << "    delete " << bufName << ";\n";

        // build the continutation specification for starting
        // has a refnum, needs to be saved in the trigger
        if (e->intExpr) {
          continutationSpec << "    c->entries.push_back(" << e->entryPtr->entryNum << ");\n";
          continutationSpec << "    c->refnums.push_back(refnum_" << cur << ");\n";
        } else {
          continutationSpec << "    c->anyEntries.push_back(" << e->entryPtr->entryNum << ");\n";
        }

        // buffers attached that we should ignore when trying to match a logical
        // AND condition
        if (entryLen > cur + 1) {
          haveAllBuffersCond << " && ";
          defs << "  ignore.insert(" << bufName << ");\n";
        }
      }
    }

    // decide based on whether buffers are found for each entry on the when
    defs << "  if (" << haveAllBuffersCond << ") {\n";

    // remove all messages fetched from SDAG buffers
    defs << removeMessagesIfFound;

    // make call to next method
    defs << "    ";

    if (constructs && !constructs->empty())
      generateCallNew(defs, encapState, encapStateChild, constructs->front()->label);
    else
      generateCallNew(defs, encapState, encapStateChild, label, "_end");

    // delete all buffered messages now that they are not needed
    defs << deleteMessagesIfFound;

    // remove the current speculative state for case statements
    if (speculativeState)
      defs << "    __dep->removeAllSpeculationIndex(" << speculativeState->name << "->speculationIndex);\n";
  
    defs << "    return 0;\n";
    defs << "  } else {\n";
    // did not find matching buffers, create a continuation

    defs << "    SDAG::Continuation* c = new SDAG::Continuation(" << nodeNum << ");\n";

    // iterate through current closures and save in a continuation
    {
      int cur = 0;
      for (list<EncapState*>::iterator iter = encapState.begin(); iter != encapState.end(); ++iter, ++cur) {
        EncapState& state = **iter;
        defs << "    c->addClosure(";

        // if the current state param is a message, create a thin wrapper for it
        // (MsgClosure) for migration purposes
        if (state.isMessage) defs << "new SDAG::MsgClosure(";
        state.name ? (defs << *state.name) : (defs << "gen" << cur);
        if (state.isMessage) defs << ")";
        defs << ");\n";
      }
    }

    // save the continutation spec for restarting this context
    defs << continutationSpec;

    // register the newly formed continutation with the runtime
    defs << "    __dep->reg(c);\n";

    // return the continuation that was just created
    defs << "    return c;\n";
    defs << "  }\n";

    endMethod(defs);

    /**
     *   Generate the ending of this 'when' clause, which calls the next in the
     *   sequence and handling deallocation of messages
     */

    // generate the _end variant of this method
    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);

    // decrease the reference count of any message state parameters
    // that are going out of scope

    // first unravel the closures so the message names are correspond to the
    // state variable names
    {
      int indent = unravelClosuresBegin(defs, true);

      // call CmiFree on each state variable going out of scope that is a message
      // (i.e. the ones that are currently brought in scope by the current
      // EntryList
      for (EntryList *el = elist; el != NULL; el = el->next, cur++) {
        if (el->entry->param->isMessage() == 1) {
          CStateVar*& sv = *el->entry->stateVars.begin();
          indentBy(defs, indent);
          defs << "CmiFree(UsrToEnv(" << sv->name << "));\n";
        }
      }
    }

    unravelClosuresEnd(defs, true);

    // generate call to the next in the sequence
    defs << "  ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");

    endMethod(defs);

    generateChildrenCode(decls, defs, entry);
  }

  void SdagConstruct::generateWhile(XStr& decls, XStr& defs, Entry* entry) {
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    defs << "    if (" << con1->text << ") {\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    } else {\n";
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";
    endMethod(defs);

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
    defs << "    if (" << con1->text << ") {\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    } else {\n";
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";
    endMethod(defs);
  }

  void SdagConstruct::generateFor(XStr& decls, XStr& defs, Entry* entry) {
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());

    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
#endif

    unravelClosuresBegin(defs);

    defs << "    " << con1->text << ";\n";
    //Record only the beginning for FOR
#if CMK_BIGSIM_CHARM
    generateEventBracket(defs, SFOR);
#endif
    defs << "    if (" << con2->text << ") {\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    } else {\n";
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";

    unravelClosuresEnd(defs);

    endMethod(defs);

    // trace
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
#endif
    unravelClosuresBegin(defs);

    defs << "   " << con3->text << ";\n";
    defs << "    if (" << con2->text << ") {\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    } else {\n";
#if CMK_BIGSIM_CHARM
    generateEventBracket(defs, SFOR_END);
#endif
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";

    unravelClosuresEnd(defs);

    endMethod(defs);
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

  void SdagConstruct::generateIf(XStr& decls, XStr& defs, Entry* entry) {
    strcpy(nameStr,label->charstar());
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);

#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SIF);
#endif

    unravelClosuresBegin(defs);

    defs << "    if (" << con1->text << ") {\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    } else {\n";
    defs << "      ";
    if (con2 != 0) {
      generateCallNew(defs, encapStateChild, encapStateChild, con2->label);
    } else {
      generateCallNew(defs, encapStateChild, encapStateChild, label, "_end");
    }
    defs << "    }\n";

    unravelClosuresEnd(defs);

    endMethod(defs);

    strcpy(nameStr,label->charstar());
    strcat(nameStr,"_end");
    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs,SIF_END);
#endif
    defs << "    ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void SdagConstruct::generateElse(XStr& decls, XStr& defs, Entry* entry) {
    strcpy(nameStr,label->charstar());
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    // trace
    generateBeginTime(defs);
    generateEventBracket(defs, SELSE);
    defs << "    ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    endMethod(defs);

    // trace
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");
    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs,SELSE_END);
#endif
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void SdagConstruct::generateForall(XStr& decls, XStr& defs, Entry* entry) {
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    defs << "    int __first = (" << con2->text << "), __last = (" << con3->text
         << "), __stride = (" << con4->text << ");\n";
    defs << "    if (__first > __last) {\n";
    defs << "      int __tmp=__first; __first=__last; __last=__tmp;\n";
    defs << "      __stride = -__stride;\n";
    defs << "    }\n";
    defs << "    SDAG::CCounter *" << counter << " = new SDAG::CCounter(__first,__last,__stride);\n";
    defs << "    for(int " << con1->text << "=__first;" << con1->text << "<=__last;"
         << con1->text << "+=__stride) {\n";
    defs << "      SDAG::ForallClosure* " << con1->text << "_cl = new SDAG::ForallClosure(" << con1->text << ");\n";
    defs << "      ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    defs << "    }\n";
    endMethod(defs);

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
    defs << "    " << counter << "->decrement(); /* DECREMENT 1 */ \n";
    defs << "    " << con1->text << "_cl->deref();\n";
    defs << "    if (" << counter << "->isDone()) {\n";
    defs << "      " << counter << "->deref();\n";
    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";
    endMethod(defs);
  }

  void SdagConstruct::generateOlist(XStr& decls, XStr& defs, Entry* entry) {
    SdagConstruct *cn;
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    defs << "    SDAG::CCounter *" << counter << "= new SDAG::CCounter(" <<
      (int)constructs->size() << ");\n";

    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it) {
      defs << "    ";
      generateCallNew(defs, encapStateChild, encapStateChild, (*it)->label);
    }
    endMethod(defs);

    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");
#if CMK_BIGSIM_CHARM
    defs << "  CkVec<void*> " <<label << "_bgLogList;\n";
#endif

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    defs << "    " <<label << "_bgLogList.insertAtEnd(_bgParentLog);\n";
#endif
    //Accumulate all the bgParent pointers that the calling when_end functions give
    defs << "    " << counter << "->decrement();\n";
 
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    defs << "    olist_" << counter << "_PathMergePoint.updateMax(currentlyExecutingPath);  /* Critical Path Detection FIXME: is the currently executing path the right thing for this? The duration ought to have been added somewhere. */ \n";
#endif

    defs << "    if (" << counter << "->isDone()) {\n";

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    defs << "      currentlyExecutingPath = olist_" << counter << "_PathMergePoint; /* Critical Path Detection */ \n";
    defs << "      olist_" << counter << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

    defs << "    " << counter << "->deref();\n";

#if CMK_BIGSIM_CHARM
    generateListEventBracket(defs, SOLIST_END);
    defs << "       "<< label <<"_bgLogList.length()=0;\n";
#endif

    defs << "      ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    defs << "    }\n";
    endMethod(defs);
  }

  void SdagConstruct::generateOverlap(XStr& decls, XStr& defs, Entry* entry) {
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SOVERLAP);
#endif
    defs << "    ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);
    endMethod(defs);

    // trace
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");
    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SOVERLAP_END);
#endif
    defs << "    ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void SdagConstruct::generateCaseList(XStr& decls, XStr& defs, Entry* entry) {
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    defs << "    SDAG::CSpeculator* " << counter << " = new SDAG::CSpeculator(__dep->getAndIncrementSpeculationIndex());\n";
  
    defs << "    SDAG::Continuation* c = 0;\n";
    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it) {
      defs << "    c = ";
      generateCallNew(defs, encapStateChild, encapStateChild, (*it)->label);
      defs << "    if (!c) return;\n";
      defs << "    else c->speculationIndex = " << counter << "->speculationIndex;\n";
    }
    endMethod(defs);

    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);

    defs << "    " << counter << "->deref();\n";
    defs << "    ";
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void SdagConstruct::generateSlist(XStr& decls, XStr& defs, Entry* entry) {
    buildTypes(encapState);
    buildTypes(encapStateChild);

    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
    defs << "    ";
    generateCallNew(defs, encapState, encapState, constructs->front()->label);
    endMethod(defs);

    generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
    defs << "    ";
    generateCallNew(defs, encapState, encapStateChild, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void SdagConstruct::generateSdagEntry(XStr& decls, XStr& defs, Entry *entry) {
    buildTypes(encapState);
    buildTypes(encapStateChild);

    if (entry->isConstructor()) {
      std::cerr << cur_file << ":" << entry->getLine()
                << ": Chare constructor cannot be defined with SDAG code" << std::endl;
      exit(1);
    }

    decls << "public:\n";

    XStr signature;

    signature << con1->text;
    signature << "(";
    if (stateVars) {
      int count = 0;
      for (list<CStateVar*>::iterator iter = stateVars->begin(); iter != stateVars->end(); ++iter) {
        CStateVar& var = **iter;
        if (var.isVoid != 1) {
          if (count != 0) signature << ", ";
          if (var.type != 0) signature << var.type << " ";
          if (var.arrayLength != NULL) signature << "* ";
          if (var.name != 0) signature << var.name;
          count++;
        }
      }
    }
    signature << ")";

    decls << "  void " <<  signature << ";\n";

    // generate wrapper for local calls to the function
    if (entry->paramIsMarshalled() || entry->param->isVoid()) {
      templateGuardBegin(false, defs);
      defs << entry->getContainer()->tspec() << " void " << entry->getContainer()->baseName() << "::" << signature << "{\n";
      defs << "  " << *entry->genClosureTypeNameProxyTemp << "*" <<
        " genClosure = new " << *entry->genClosureTypeNameProxyTemp << "()" << ";\n";
      if (stateVars) {
        int i = 0;
        for (list<CStateVar*>::iterator it = stateVars->begin(); it != stateVars->end(); ++it, ++i) {
          CStateVar& var = **it;
          if (var.name) {
            if (i == 0 && *var.type == "int")
              defs << "  genClosure->__refnum" << " = " << var.name << ";\n";
            else if (i == 0)
              defs << "  genClosure->__refnum" << " = 0;\n";
            defs << "  genClosure->" << var.name << " = " << var.name << ";\n";
          }
        }
      }

      defs << "  " << con1->text << "(genClosure);\n";
      defs << "  genClosure->deref();\n";
      defs << "}\n\n";
      templateGuardEnd(defs);
    }

    generateSignatureNew(decls, defs, entry, false, "void", con1->text, false, encapState);

#if CMK_BIGSIM_CHARM
    generateEndSeq(defs);
#endif
    if (!entry->getContainer()->isGroup() || !entry->isConstructor())
      generateTraceEndCall(defs);

    defs << "  if (!__dep.get())\n"
         << "      _sdag_init();\n";

    // is a message sdag entry, in this case since this is a SDAG entry, there
    // will only be one parameter which is the message (called 'gen0')
    if (!entry->paramIsMarshalled() && !entry->param->isVoid()) {
      // increase reference count by one for the state parameter
      defs << "  CmiReference(UsrToEnv(gen0));\n";
    }

    defs << "  ";
    generateCallNew(defs, encapStateChild, encapStateChild, constructs->front()->label);

#if CMK_BIGSIM_CHARM
    generateTlineEndCall(defs);
    generateBeginExec(defs, "spaceholder");
#endif
    if (!entry->getContainer()->isGroup() || !entry->isConstructor())
      generateDummyBeginExecute(defs);

    endMethod(defs);

    decls << "private:\n";
    generateSignatureNew(decls, defs, entry, false, "void", con1->text, true,
#if CMK_BIGSIM_CHARM
                         encapStateChild
#else
                         encapState
#endif
                         );

    if (!entry->paramIsMarshalled() && !entry->param->isVoid()) {
      // decrease reference count by one for the message
      defs << "  CmiFree(UsrToEnv(gen0));\n";
    }

    endMethod(defs);
  }

  void AtomicConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);

    int indent = unravelClosuresBegin(defs);

    indentBy(defs, indent);
    defs << "{ // begin serial block\n";
    defs << text << "\n";
    indentBy(defs, indent);
    defs << "} // end serial block\n";

    unravelClosuresEnd(defs);

    indentBy(defs, 1);
    generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);
  }

  void generateSignature(XStr& str,
                         const XStr* name, const char* suffix,
                         list<CStateVar*>* params) {
    
  }
  void generateSignature(XStr& decls, XStr& defs,
                         const Entry* entry, bool declareStatic, const char* returnType,
                         const XStr* name, bool isEnd,
                         list<CStateVar*>* params) {
    generateSignature(decls, defs, entry->getContainer(), declareStatic, returnType,
                      name, isEnd, params);
  }
  void generateSignature(XStr& decls, XStr& defs,
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

  void generateSignatureNew(XStr& decls, XStr& defs,
                            const Entry* entry, bool declareStatic, const char* returnType,
                            const XStr* name, bool isEnd,
                            list<EncapState*> encap) {
    generateSignatureNew(decls, defs, entry->getContainer(), declareStatic, returnType,
                         name, isEnd, encap);
  }
  void generateSignatureNew(XStr& decls, XStr& defs, const Chare* chare,
                            bool declareStatic, const char* returnType,
                            const XStr* name, bool isEnd, list<EncapState*> encap) {
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

    op << ")";

    decls << op << ";\n";
    defs << op << " {\n";
  }

  void SdagConstruct::generateCallNew(XStr& op, list<EncapState*>& scope,
                                      list<EncapState*>& next, const XStr* name,
                                      const char* nameSuffix) {
    op << name << (nameSuffix ? nameSuffix : "") << "(";

    int cur = 0;
    for (list<EncapState*>::iterator iter = next.begin(); iter != next.end(); ++iter, ++cur) {
      EncapState *state = *iter;

      if (state->type) {
        if (cur > scope.size() - 1) {
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

  void AtomicConstruct::generateTrace() {
    char traceText[1024];
    if (traceName) {
      sprintf(traceText, "%s_%s", CParsedFile::className->charstar(), traceName->charstar());
      // remove blanks
      for (unsigned int i=0; i<strlen(traceText); i++)
        if (traceText[i]==' '||traceText[i]=='\t') traceText[i]='_';
    }
    else {
      sprintf(traceText, "%s%s", CParsedFile::className->charstar(), label->charstar());
    }
    traceName = new XStr(traceText);

    if (con1) con1->generateTrace();
  }

  void SdagConstruct::generateTraceBeginCall(XStr& op) {
    if(traceName)
      op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (" << "_sdag_idx_" << traceName << "()), CkMyPe(), 0, NULL); \n";
  }

  void SdagConstruct::generateDummyBeginExecute(XStr& op) {
    op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _dummyEP, CkMyPe(), 0, NULL); \n";
  }

  void SdagConstruct::generateTraceEndCall(XStr& op) {
    op << "    " << "_TRACE_END_EXECUTE(); \n";
  }

  void SdagConstruct::generateBeginExec(XStr& op, const char *name) {
    op << "     " << "_TRACE_BG_BEGIN_EXECUTE_NOMSG(\""<<name<<"\",&_bgParentLog,1);\n"; 
  }

  void SdagConstruct::generateEndExec(XStr& op){
    op << "     " << "_TRACE_BG_END_EXECUTE(0);\n";
  }

  void SdagConstruct::generateBeginTime(XStr& op) {
    //Record begin time for tracing
    op << "    double __begintime = CkVTimer(); \n";
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
    generateTraceEndCall(op);
    generateEndExec(op);
  }

  void SdagConstruct::generateEventBracket(XStr& op, int eventType) {
    (void) eventType;
    //Trace this event
    op << "     _TRACE_BG_USER_EVENT_BRACKET(\"" << nameStr
       << "\", __begintime, CkVTimer(),&_bgParentLog); \n";
  }

  void SdagConstruct::generateListEventBracket(XStr& op, int eventType) {
    (void) eventType;
    op << "    _TRACE_BGLIST_USER_EVENT_BRACKET(\"" << nameStr
       << "\",__begintime,CkVTimer(),&_bgParentLog, " << label
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
      generateSignature(decls, defs, chare, true, "int", &idxName, false, NULL);
      defs << "  static int epidx = " << regName << "();\n"
           << "  return epidx;\n";
      endMethod(defs);

      generateSignature(decls, defs, chare, true, "int", &regName, false, NULL);
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


  void RemoveSdagComments(char *str) {
    char *ptr = str;
    while ((ptr = strstr(ptr, "//"))) {
      char *lend = strstr(ptr, "\n");
      if (lend==NULL) break;
      while (ptr != lend) *ptr++=' ';
    }
  }
}
