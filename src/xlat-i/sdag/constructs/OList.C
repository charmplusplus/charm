#include "OList.h"
#include "CParsedFile.h"
#include <list>

namespace xi {

OListConstruct::OListConstruct(SdagConstruct *single_construct)
: SdagConstruct(SOLIST, single_construct)
{
  label_str = "olist";
}

OListConstruct::OListConstruct(SdagConstruct *single_construct, SListConstruct *tail)
: SdagConstruct(SOLIST, single_construct, tail)
{
  label_str = "olist";
}

void OListConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
  defs << "  SDAG::CCounter *" << counter << "= new SDAG::CCounter(" <<
    (int)constructs->size() << ");\n";

  for (std::list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
       ++it) {
    defs << "  ";
    generateCall(defs, encapStateChild, encapStateChild, (*it)->label);
  }
  endMethod(defs);

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
#if CMK_BIGSIM_CHARM
  defs << "  CkVec<void*> " <<label << "_bgLogList;\n";
#endif

  generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  defs << "  " <<label << "_bgLogList.insertAtEnd(_bgParentLog);\n";
#endif
  //Accumulate all the bgParent pointers that the calling when_end functions give
  defs << "  " << counter << "->decrement();\n";

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "  olist_" << counter << "_PathMergePoint.updateMax(currentlyExecutingPath);  /* Critical Path Detection FIXME: is the currently executing path the right thing for this? The duration ought to have been added somewhere. */ \n";
#endif

  defs << "  if (" << counter << "->isDone()) {\n";

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "    currentlyExecutingPath = olist_" << counter << "_PathMergePoint; /* Critical Path Detection */ \n";
  defs << "    olist_" << counter << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  defs << "  " << counter << "->deref();\n";

#if CMK_BIGSIM_CHARM
  generateListEventBracket(defs, SOLIST_END);
  defs << "    " << label <<"_bgLogList.length()=0;\n";
#endif

  defs << "    ";
  generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  defs << "  }\n";
  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void OListConstruct::numberNodes() {
  nodeNum = numOlists++;
  SdagConstruct::numberNodes();
}

void OListConstruct::propagateState(std::list<EncapState*> encap,
                                    std::list<CStateVar*>& plist,
                                    std::list<CStateVar*>& wlist,
                                    int uniqueVarNum) {
  CStateVar *sv;

  stateVars = new std::list<CStateVar*>();

  encapState = encap;

  stateVarsChildren = new std::list<CStateVar*>(plist);
  stateVars->insert(stateVars->end(), plist.begin(), plist.end());
  {
    char txt[128];
    sprintf(txt, "_co%d", nodeNum);
    counter = new XStr(txt);
    sv = new CStateVar(0, "SDAG::CCounter *", 0, txt, 0, NULL, 1);
    sv->isCounter = true;
    stateVarsChildren->push_back(sv);

    std::list<CStateVar*> lst;
    lst.push_back(sv);
    EncapState *state = new EncapState(NULL, lst);
    state->type = new XStr("SDAG::CCounter");
    state->name = new XStr(txt);
    encap.push_back(state);
  }

  encapStateChild = encap;

  propagateStateToChildren(encap, *stateVarsChildren, wlist, uniqueVarNum);
}

}   // namespace xi
