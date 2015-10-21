#include "When.h"
#include "xi-Chare.h"

using std::list;

namespace xi {

WhenConstruct::WhenConstruct(EntryList *el, SdagConstruct *body)
: BlockConstruct(SWHEN, 0, 0, 0, 0, 0, body, el)
, speculativeState(0)
{
  label_str = "when";
}

  void WhenConstruct::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen) {
    elist->generateEntryList(CEntrylist, this);  /* con1 is the WHEN's ELIST */
    generateChildrenEntryList(CEntrylist, thisWhen);
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
    int cntr = 0;
    bool dummy_var = false;
    while (el != NULL) {
      pl = el->entry->param;
      if (!pl->isVoid()) {
        while(pl != NULL) {
          if (pl->getGivenName() == NULL){//if the parameter doesn't have a name, generate a dummy name
            char s[128];
            sprintf(s, "gen_name%d", cntr);
            pl->setGivenName(s);
            cntr++;
            dummy_var = true;
          }
          sv = new CStateVar(pl);
          if(!dummy_var){ //only if it's not a dummy variable, propagate it to the children 
            stateVarsChildren->push_back(sv);
            whensEntryMethodStateVars.push_back(sv);
          }
          else dummy_var = false;
          whenCurEntry.push_back(sv);
          el->entry->addEStateVar(sv);
          pl = pl->next;
        }
      }

      EncapState* state = new EncapState(el->entry, whenCurEntry);
      if (!el->entry->paramIsMarshalled() && !el->entry->param->isVoid())
        state->isMessage = true;
      if (!el->entry->param->isVoid())
	encap.push_back(state);
      whenCurEntry.clear();
      el = el->next;
    }

    encapStateChild = encap;

    propagateStateToChildren(encap, *stateVarsChildren, whensEntryMethodStateVars, uniqueVarNum);
  }

  void WhenConstruct::generateWhenCode(XStr& op, int indent) {
    buildTypes(encapState);
    buildTypes(encapStateChild);

    // generate the call for this when

#if CMK_BIGSIM_CHARM
    // bgLog2 stores the parent dependence of when, e.g. for, olist
    indentBy(op, indent);
    op << "cmsgbuf->bgLog2 = (void*)static_cast<SDAG::TransportableBigSimLog*>(c->closure[1])->log;\n";
#endif

    // output the when function's name
    indentBy(op, indent);
    op << this->label << "(";

    // output all the arguments to the function that are stored in a continuation
    int cur = 0;
    for (list<EncapState*>::iterator iter = encapState.begin();
         iter != encapState.end(); ++iter, ++cur) {
      EncapState& state = **iter;
      op << "\n";
      indentBy(op, indent + 1);
      if (state.isMessage)
        op << "static_cast<" << *state.type << "*>(static_cast<SDAG::MsgClosure*>(c->closure[" << cur << "])->msg)";
      else if (state.isBgParentLog)
        op << "NULL";
      else
        op << "static_cast<" << *state.type << "*>(c->closure[" << cur << "])";
      if (cur != encapState.size() - 1) op << ", ";
    }

    int prev = cur;

    cur = 0;
    for (EntryList *el = elist; el != NULL; el = el->next, cur++)
      if (el->entry->intExpr) {
        if ((cur+prev) > 0) op << ", ";
	op << "\n";
        indentBy(op, indent + 1);
        op << "c->refnums[" << cur << "]";
      }

    op << "\n";
    indentBy(op, indent);
    op << ");\n";
#if CMK_BIGSIM_CHARM
    generateTlineEndCall(op);
    generateBeginExec(op, "sdagholder");
#endif
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

    int entryLen = 0, numRefs = 0;

    // count the number of entries this when contains (for logical ands) and
    // the number of reference numbers
    {
      int cur = 0;
      for (EntryList *el = elist; el != NULL; el = el->next, cur++) {
        entryLen++;
        if (el->entry->intExpr) numRefs++;
      }
    }

    // if reference numbers exist for this when, generate a wrapper that calls
    // the when method with the reference numbers determined based on the
    // current state
    if (numRefs > 0) {
      sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
      generateClosureSignature(decls, defs, entry, false, "SDAG::Continuation*", label, false, encapState);

      // if we have a reference number in the closures, we need to unravel the state
      int cur = 0;
      for (EntryList *el = elist; el != NULL; el = el->next, cur++)
        if (el->entry->intExpr) defs << "  CMK_REFNUM_TYPE refnum_" << cur << ";\n";
      int indent = 2;

      // unravel the closures so the potential refnum expressions can be resolved
      indent = unravelClosuresBegin(defs);
      indentBy(defs, indent);
      // create a new scope for unraveling the closures
      defs << "{\n";

      cur = 0;
      // generate each refnum variable we need that can access the internal closure state
      for (EntryList *el = elist; el != NULL; el = el->next, cur++)
        if (el->entry->intExpr) {
          indentBy(defs, indent + 1);
          defs << "refnum_" << cur << " = " << (el->entry->intExpr ? el->entry->intExpr : "0") << ";\n";
        }

      // end the unraveling of closures
      indentBy(defs, indent);
      defs << "}\n";
      unravelClosuresEnd(defs);

      // generate the call to the actual when that takes the reference numbers as arguments
      defs << "  return " << label << "(";
      cur = 0;
      for (list<EncapState*>::iterator iter = encapState.begin(); iter != encapState.end(); ++iter, ++cur) {
        EncapState *state = *iter;
        if (state->name) defs << *state->name; else defs << "gen" << cur;
        if (cur != encapState.size() - 1) defs << ", ";
      }
      for (int i = 0; i < numRefs; i++) defs << ((cur+i) > 0 ? ", " : "") << "refnum_" << i;
      defs << ");\n";

      endMethod(defs);
    }

    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    generateClosureSignature(decls, defs, entry, false, "SDAG::Continuation*", label, false, encapState, numRefs);

#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
#endif

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
             << ", " << (entryLen > 1 ? "&ignore" : "0") // the ignore set
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
          defs << "  if (" << bufName << ") ignore.insert(" << bufName << ");\n";
        }
      }
    }

    // decide based on whether buffers are found for each entry on the when
    defs << "  if (" << haveAllBuffersCond << ") {\n";

#if CMK_BIGSIM_CHARM
    {
      // TODO: instead of this, add a length field to EntryList
      defs << "    void* logs1["<< entryLen << "]; \n";
      defs << "    void* logs2["<< entryLen + 1 << "]; \n";
      int localnum = 0;
      int cur = 0;
      for (EntryList *el = elist; el != NULL; el = el->next, cur++) {
        XStr bufName("buf");
        bufName << cur;
        defs << "    logs1[" << localnum << "] = " << bufName << "->bgLog1; \n";
        defs << "    logs2[" << localnum << "] = " << bufName << "->bgLog2; \n";
        localnum++;
      }
      defs << "    logs2[" << localnum << "] = " << "_bgParentLog; \n";
      generateEventBracket(defs, SWHEN);
      defs << "    _TRACE_BG_FORWARD_DEPS(logs1, logs2, "<< localnum << ", _bgParentLog);\n";
    }
#endif

    // remove all messages fetched from SDAG buffers
    defs << removeMessagesIfFound;

    // remove the current speculative state for case statements
    if (speculativeState)
      defs << "    __dep->removeAllSpeculationIndex(" << speculativeState->name << "->speculationIndex);\n";

    // make call to next method
    defs << "    ";

    if (constructs && !constructs->empty())
      generateCall(defs, encapState, encapStateChild, constructs->front()->label);
    else
      generateCall(defs, encapState, encapStateChild, label, "_end");

    // delete all buffered messages now that they are not needed
    defs << deleteMessagesIfFound;

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
        if (state.isBgParentLog) defs << "new SDAG::TransportableBigSimLog(";
        state.name ? (defs << *state.name) : (defs << "gen" << cur);
        if (state.isMessage || state.isBgParentLog) defs << ")";
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
    generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);

#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SWHEN_END);
#endif

    // decrease the reference count of any message state parameters
    // that are going out of scope

    // first check if we have any messages going out of scope
    bool messageOutOfScope = false;
    int cur = 0;
    for (EntryList *el = elist; el != NULL; el = el->next, cur++)
      if (el->entry->param->isMessage() == 1)
        messageOutOfScope = true;

    // first unravel the closures so the message names are correspond to the
    // state variable names
    if (messageOutOfScope) {
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

      unravelClosuresEnd(defs, true);
    }

    // generate call to the next in the sequence
    defs << "  ";
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");

    endMethod(defs);

    generateChildrenCode(decls, defs, entry);
  }

void WhenConstruct::numberNodes() {
  nodeNum = numWhens++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
