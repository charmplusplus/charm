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

SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1)
{
  con1 = 0;  con2 = 0; con3 = 0; con4 = 0;
  type = t;
  traceName=NULL;
  publishesList = new list<SdagConstruct*>();
  constructs = new list<SdagConstruct*>();
  constructs->push_back(construct1);
}

SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList)
{
  con1=0; con2=0; con3=0; con4=0;
  type = t;
  traceName=NULL;
  publishesList = new list<SdagConstruct*>();
  constructs = new list<SdagConstruct*>();
  constructs->push_back(construct1);
  constructs->insert(constructs->end(), aList->constructs->begin(), aList->constructs->end());
}

SdagConstruct::SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
			     SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el)
{
  text = txt;
  type = t;
  traceName=NULL;
  con1 = c1; con2 = c2; con3 = c3; con4 = c4;
  publishesList = new list<SdagConstruct*>();
  constructs = new list<SdagConstruct*>();
  if (constructAppend != 0) {
    constructs->push_back(constructAppend);
  }
  elist = el;
}

SdagConstruct::SdagConstruct(EToken t, const char *entryStr, const char *codeStr, ParamList *pl)
{
  type = t;
  traceName=NULL;
  text = new XStr(codeStr);
  connectEntry = new XStr(entryStr);
  con1 = 0; con2 = 0; con3 = 0; con4 =0;
  publishesList = new list<SdagConstruct*>();
  constructs = new list<SdagConstruct*>();
  param = pl;
}

void SdagConstruct::numberNodes(void)
{
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
    case SFORWARD: nodeNum = numForwards++; break;
    case SCONNECT: nodeNum = numConnects++; break;
    case SCASE: nodeNum = numCases++; break;
    case SCASELIST: nodeNum = numCaseLists++; break;
    case SINT_EXPR:
    case SIDENT: 
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::numberNodes));
  }
}

void SdagConstruct::labelNodes(void)
{
  char text[128];
  switch(type) {
    case SSDAGENTRY:
      sprintf(text, "%s", con1->text->charstar());
      label = new XStr(text);
      break;
    case SOVERLAP: 
      sprintf(text, "_overlap_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SWHEN: 
      sprintf(text, "_when_%d", nodeNum); 
      label = new XStr(text);
      EntryList *el;
      el = elist;
      while (el !=NULL) {
        el->entry->label = new XStr(el->entry->name);
        el=el->next; 
      }
      break;
    case SFOR: 
      sprintf(text, "_for_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SWHILE: 
      sprintf(text, "_while_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SIF: 
      sprintf(text, "_if_%d", nodeNum); 
      label = new XStr(text);
      if(con2!=0) con2->labelNodes();
      break;
    case SELSE: 
      sprintf(text, "_else_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SFORALL: 
      sprintf(text, "_forall_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SSLIST: 
      sprintf(text, "_slist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SOLIST: 
      sprintf(text, "_olist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SATOMIC: 
      sprintf(text, "_atomic_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SFORWARD: 
      sprintf(text, "_forward_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SCONNECT:
      sprintf(text, "_connect_%s",connectEntry->charstar()); 
      label = new XStr(text);
      break;
    case SCASE:
      sprintf(text, "_case_%d", nodeNum);
      label = new XStr(text);
      break;
    case SCASELIST:
      sprintf(text, "_caselist_%d", nodeNum);
      label = new XStr(text);
      break;
    case SINT_EXPR:
    case SIDENT:
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::labelNodes));
  }
}

void EntryList::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen)
{
   EntryList *el;
   el = this;
   while (el != NULL)
   {
     el->entry->generateEntryList(CEntrylist, thisWhen);
     el = el->next;
   }
}

void Entry::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen)
{
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

void SdagConstruct::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen)
{
  if (SIF == type && con2 != 0)
    con2->generateEntryList(CEntrylist, thisWhen);
  generateChildrenEntryList(CEntrylist, thisWhen);
}

void WhenConstruct::generateEntryList(list<CEntry*>& CEntrylist, WhenConstruct *thisWhen)
{
  elist->generateEntryList(CEntrylist, this);  /* con1 is the WHEN's ELIST */
  generateChildrenEntryList(CEntrylist, thisWhen);
}

void SdagConstruct::generateChildrenEntryList(list<CEntry*>& CEntrylist,
                                              WhenConstruct *thisWhen) {
  if (constructs != 0) {
    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it)
      (*it)->generateEntryList(CEntrylist, thisWhen);
  }
}

void SdagConstruct::generateConnectEntries(XStr& decls) {
   decls << "  void " <<connectEntry << "(";
   ParamList *pl = param;
   XStr msgParams;
   if (pl->isVoid() == 1) {
     decls << "void";
   }
   else if (pl->isMessage() == 1){
     decls << pl->getBaseName() <<" *" <<pl->getGivenName();
   }
   else {
    decls << "CkMarshallMsg *" /*<< connectEntry*/ <<"_msg";
   }
   decls << ") {\n";

   if (!pl->isVoid() && !pl->isMessage()) {
    msgParams <<"   char *impl_buf= _msg->msgBuf;\n";
    param->beginUnmarshall(msgParams);
   }

   decls << msgParams <<"\n";
   decls << "  " <<text <<"\n";

   decls << "  }\n";
}

void SdagConstruct::generateConnectEntryList(list<SdagConstruct*>& ConnectEList) {
  if (type == SCONNECT)
     ConnectEList.push_back(this);
  if (constructs != 0) {
    for (list<SdagConstruct*>::iterator iter = constructs->begin(); iter != constructs->end(); ++iter)
      (*iter)->generateConnectEntryList(ConnectEList);
  }
}

void SdagConstruct::propagateState(int uniqueVarNum)
{ 
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
    
    encap.push_back(new EncapState(this->entry, *stateVars));
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
    (*it)->propagateState(encap, *stateVarsChildren, whensEntryMethodStateVars, *publishesList, uniqueVarNum);
}

void SdagConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist, list<SdagConstruct*>& publist, int uniqueVarNum)
{
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
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "CCounter *", 0, txt, 0, NULL, 1);
        sv->isCounter = true;
        stateVarsChildren->push_back(sv);
      }
      break;
    case SIF:
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      stateVarsChildren = stateVars;
      if(con2 != 0) con2->propagateState(encap, plist, wlist, publist, uniqueVarNum);
      break;
    case SCASELIST:
      stateVarsChildren = new list<CStateVar*>(plist);
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      {
        char txt[128];
        sprintf(txt, "_cs%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "CSpeculator *", 0, txt, 0, NULL, 1);
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
        state->type = new XStr("CSpeculator");
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
        sv = new CStateVar(0, "CCounter *", 0, txt, 0, NULL, 1);
        sv->isCounter = true;
        stateVarsChildren->push_back(sv);

        list<CStateVar*> lst;
        lst.push_back(sv);
        EncapState *state = new EncapState(NULL, lst);
        state->type = new XStr("CCounter");
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
    case SFORWARD:
      stateVarsChildren = new list<CStateVar*>(wlist);
      stateVars->insert(stateVars->end(), plist.begin(), plist.end());
      break;
    case SCONNECT: 
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

  propagateStateToChildren(encap, *stateVarsChildren, wlist, publist, uniqueVarNum);
  delete whensEntryMethodStateVars;
}

void WhenConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist, list<SdagConstruct*>& publist, int uniqueVarNum) {
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
    if (pl->isVoid()) {
      sv = new CStateVar(1, NULL, 0, NULL, 0, NULL, 0);
      //stateVars->push_back(sv);
      stateVarsChildren->push_back(sv);
      whensEntryMethodStateVars.push_back(sv);
      whenCurEntry.push_back(sv);
      el->entry->addEStateVar(sv);
    }
    else {
      while(pl != NULL) {
        sv = new CStateVar(pl);
        stateVarsChildren->push_back(sv);
        whensEntryMethodStateVars.push_back(sv);
        whenCurEntry.push_back(sv);
        el->entry->addEStateVar(sv);

        pl = pl->next;
      }
    }
    encap.push_back(new EncapState(el->entry, whenCurEntry));
    whenCurEntry.clear();
    el = el->next;
  }

  encapStateChild = encap;

  propagateStateToChildren(encap, *stateVarsChildren, whensEntryMethodStateVars, publist, uniqueVarNum);
}


void AtomicConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist, list<SdagConstruct*>& publist, int uniqueVarNum) {
  stateVars = new list<CStateVar*>();
  stateVars->insert(stateVars->end(), plist.begin(), plist.end());
  stateVarsChildren = stateVars;

  encapState = encap;
  encapStateChild = encap;

  if (con1 != 0) {
    publist.push_back(con1);
    /*SdagConstruct *sc;
    SdagConstruct *sc1;
    for(sc =publist.begin(); !publist.end(); sc=publist.next()) {
       for(sc1=sc->constructs->begin(); !sc->constructs->end(); sc1 = sc->constructs->next())
       printf("Publist = %s\n", sc1->text->charstar());
    }*/
  }
}

void SdagConstruct::propagateStateToChildren(list<EncapState*> encap, list<CStateVar*>& stateVarsChildren, list<CStateVar*>& wlist, list<SdagConstruct*>& publist, int uniqueVarNum) {
  if (constructs != 0) {
    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it)
      (*it)->propagateState(encap, stateVarsChildren, wlist, publist, uniqueVarNum);
  }
}

void SdagConstruct::generateCode(XStr& decls, XStr& defs, Entry *entry)
{
  switch(type) {
    case SSDAGENTRY:
      generateSdagEntry(decls, defs, entry);
      break;
    case SSLIST:
      generateSlist(decls, defs, entry);
      break;
    case SOLIST:
      generateOlist(decls, defs, entry);
      break;
    case SFORALL:
      generateForall(decls, defs, entry);
      break;
    case SIF:
      generateIf(decls, defs, entry);
      if(con2 != 0)
        con2->generateCode(decls, defs, entry);
      break;
    case SELSE:
      generateElse(decls, defs, entry);
      break;
    case SWHILE:
      generateWhile(decls, defs, entry);
      break;
    case SFOR:
      generateFor(decls, defs, entry);
      break;
    case SCASE:
    case SOVERLAP:
      generateOverlap(decls, defs, entry);
      break;
    case SFORWARD:
      generateForward(decls, defs, entry);
      break;
    case SCONNECT:
      generateConnect(decls, defs, entry);
      break;
    case SCASELIST:
      generateCaseList(decls, defs, entry);
      break;
    default:
      break;
  }
  generateChildrenCode(decls, defs, entry);
}

void SdagConstruct::generateChildrenCode(XStr& decls, XStr& defs, Entry* entry) {
  if (constructs != 0) {
    for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
         ++it)
      (*it)->generateCode(decls, defs, entry);
  }
}

void SdagConstruct::buildTypes(list<EncapState*>& state) {
  for (list<EncapState*>::iterator iter = state.begin(); iter != state.end(); ++iter) {
    EncapState& encap = **iter;
    if (!encap.type) {
      if (encap.entry->entryPtr && encap.entry->entryPtr->decl_entry)
        encap.type = encap.entry->entryPtr->decl_entry->genStructTypeNameProxy;
      else
        encap.type = encap.entry->genStructTypeNameProxy;
    }
  }
}

void SdagConstruct::generateWhenCodeNew(XStr& op) {
  buildTypes(encapState);
  buildTypes(encapStateChild);

  // generate call this when
  op << "        " << this->label << "(";
  int cur = 0;
  for (list<EncapState*>::iterator iter = encapState.begin();
       iter != encapState.end(); ++iter, ++cur) {
    EncapState& state = **iter;
    op << "\n          reinterpret_cast<" << *state.type << "*>(t->args[" << cur << "])";
    if (cur != encapState.size() - 1) op << ", ";
  }  
  op << "\n        );\n";
}

void SdagConstruct::generateWhenCode(XStr& op)
{
  SdagConstruct *cn = this;
  XStr whenParams = "";
  int i = 0;
  int iArgs = 0;
  int generatedWhenParams = 0;
  bool lastWasVoid = false;
  bool paramMarshalling = false;

#if CMK_BIGSIM_CHARM
  // bgLog2 stores the parent dependence of when, e.g. for, olist
  op << "        cmsgbuf->bgLog2 = (void*)dynamic_cast<TransportableBigSimLog*>(tr->args[1])->log;\n";
#endif

  for (list<CStateVar*>::iterator iter = stateVars->begin();
       iter != stateVars->end();
       ++iter, ++i) {
    CStateVar *sv = *iter;
    if ((sv->isMsg == 0) && (paramMarshalling == 0) && (sv->isVoid ==0)){
      paramMarshalling =1;
      op << "        CkMarshallMsg *impl_msg" << cn->nodeNum << " = (CkMarshallMsg*)dynamic_cast<TransportableMsg*>(tr->args[" << iArgs++ << "])->msg;\n";
      op << "        char *impl_buf" <<cn->nodeNum <<"=((CkMarshallMsg *)impl_msg" <<cn->nodeNum <<")->msgBuf;\n";
      op << "        PUP::fromMem implP" <<cn->nodeNum <<"(impl_buf" <<cn->nodeNum <<");\n";
    }
    if (sv->isMsg == 1) {
      if (generatedWhenParams != 0)
        whenParams.append(", ");
#if CMK_BIGSIM_CHARM
      if(i==1) {
        whenParams.append(" NULL ");
        generatedWhenParams++;

        lastWasVoid=0;
        // skip this arg which is supposed to be _bgParentLog
        iArgs++;
        continue;
      }
#endif

      if (sv->isMsg && !sv->isCounter && !sv->isSpeculator && !sv->isBgParentLog)
        whenParams << "(" << sv->type->charstar() << ")" << "dynamic_cast<TransportableMsg*>(";
      else if (sv->isCounter)
        whenParams << "dynamic_cast<CCounter*>(";
      else if (sv->isSpeculator)
        whenParams << "dynamic_cast<CSpeculator*>(";
      else if (sv->isBgParentLog)
        whenParams << "dynamic_cast<TransportableBigSimLog*>(";

      whenParams << "tr->args[" << iArgs << "])";

      if (sv->isMsg && !sv->isCounter && !sv->isSpeculator && !sv->isBgParentLog)
        whenParams << "->msg";
      else if (sv->isMsg && sv->isBgParentLog)
        whenParams << "->log";

      generatedWhenParams++;
      iArgs++;
    }
    else if (sv->isVoid == 1) {
      op << "        if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
      op << "        tr->args[" <<iArgs++ <<"] = NULL;\n";
    } else if ((sv->isMsg == 0) && (sv->isVoid == 0)) {
      if (generatedWhenParams != 0)
        whenParams.append(", ");

      whenParams.append(*(sv->name));
      generatedWhenParams++;

      if (sv->arrayLength != 0)
        op << "        int impl_off" << cn->nodeNum << "_" << sv->name << "; implP"
          <<cn->nodeNum << "|impl_off" <<cn->nodeNum  << "_" << sv->name << ";\n";
      else
        op << "        " << sv->type << " " << sv->name << "; implP"
          <<cn->nodeNum << "|" << sv->name << ";\n";
    }
    lastWasVoid = sv->isVoid;
  }
  if (paramMarshalling == 1)
    op << "        impl_buf"<<cn->nodeNum << "+=CK_ALIGN(implP" <<cn->nodeNum <<".size(),16);\n";
  for (list<CStateVar*>::iterator iter = stateVars->begin();
       iter != stateVars->end();
       ++iter) {
    CStateVar *sv = *iter;
    if (sv->arrayLength != 0)
      op << "        " << sv->type << " *" << sv->name << "=(" << sv->type << " *)(impl_buf" <<cn->nodeNum
        << "+impl_off" <<cn->nodeNum << "_" << sv->name << ");\n";
  }
  if (paramMarshalling == 1)
    op << "        delete (CkMarshallMsg *)impl_msg" <<cn->nodeNum <<";\n";
  op << "        " << cn->label << "(" << whenParams;
  op << ");\n";
  op << "        delete tr;\n";

#if CMK_BIGSIM_CHARM
  cn->generateTlineEndCall(op);
  cn->generateBeginExec(op, "sdagholder");
#endif
  op << "    ";
  cn->generateDummyBeginExecute(op);

  op << "        return;\n";
}

void SdagConstruct::generateConnect(XStr& decls, XStr& defs, Entry* entry) {
  generateSignature(decls, defs, entry, false, "void", label, false, NULL);
  defs << "    int index;\n";
  if ((param->isVoid() == 0) && (param->isMessage() == 0)) {
     defs << "    CkMarshallMsg *x;\n";
     defs << "    index = CkIndex_Ar1::" <<connectEntry <<"(x);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  else if (param->isVoid() == 1) {
     defs << "    index = CkIndex_Ar1::" <<connectEntry <<"(void);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  else {
     defs << "    " << param->getBaseName() <<" *x;\n";  // replace
     defs << "    index = CkIndex_Ar1::" <<connectEntry <<"(x);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  defs << "    myPublish->get_" <<connectEntry <<"(cb);\n";  //replace - myPublish

  endMethod(defs);
}

void SdagConstruct::generateForward(XStr& decls, XStr& defs, Entry* entry) {
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
       ++it) {
    defs << "    { ";
    generateCall(defs, *stateVarsChildren, (*it)->text);
    defs<<" }\n";
  }
  generateCall(defs, *stateVarsChildren, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
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

void WhenConstruct::generateCodeNew(XStr& decls, XStr& defs, Entry* entry) {
  buildTypes(encapState);
  buildTypes(encapStateChild);

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateSignatureNew(decls, defs, entry, false, "SDAG::Trigger*", label, false, encapState);

  int entryLen = 0;

  {
    int cur = 0;
    for (EntryList *el = elist; el != NULL; el = el->next, cur++) entryLen++;
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
      defs << "  SDAG::Buffer* " << bufName << " = __dep->tryFindMessage("
           << e->entryPtr->entryNum // entry number
           << ", " << (e->intExpr ? "true" : "false") // has a ref number?
           << ", " << (e->intExpr ? e->intExpr : "0") // the ref number
           << ", " << (cur != 0 ? "true" : "false")   // has a ignore set?
           << (entryLen > 1 ? ", ignore" : "") // the ignore set
           << ");\n";
      haveAllBuffersCond << bufName;
      removeMessagesIfFound << "    __dep->removeMessage(" << bufName << ");\n";
      deleteMessagesIfFound << "    delete " << bufName << ";\n";

      // build the continutation specification for starting
      // has a refnum, needs to be saved in the trigger
      if (e->intExpr) {
        continutationSpec << "    t->entries.push_back(" << e->entryPtr->entryNum << ");\n";
        continutationSpec << "    t->refnums.push_back(" << e->intExpr << ");\n";
      } else {
        continutationSpec << "    t->anyEntries.push_back(" << e->entryPtr->entryNum << ");\n";
      }

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

  defs << "    SDAG::Trigger* t = new SDAG::Trigger(" << nodeNum << ");\n";

  // iterative through current state and save in a trigger
  {
    int cur = 0;
    for (list<EncapState*>::iterator iter = encapState.begin(); iter != encapState.end(); ++iter, ++cur) {
      EncapState& state = **iter;
      defs << "    t->args.push_back(";
      state.name ? (defs << *state.name) : (defs << "gen" << cur);
      defs << ");\n";
    }
  }

  // save the continutation spec for restarting this context
  defs << continutationSpec;

  // register the newly formed continutation with the runtime
  defs << "    __dep->reg(t);\n";
  defs << "    return t;\n";
  defs << "  }\n";

  endMethod(defs);

  // generate the _end variant of this method

  generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
  defs << "    ";
  generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);

}

void WhenConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry)
{
  generateCodeNew(decls, defs, entry);
  generateChildrenCode(decls, defs, entry);
  return;

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateSignature(decls, defs, entry, false, "CWhenTrigger*", label, false, stateVars);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif

  CStateVar *sv;

  Entry *e;
  EntryList *el;
  int curEntry = 0;
  el = elist;
  while (el != NULL){
    e = el->entry;
    if (e->param->isVoid() == 1)
      defs << "    CMsgBuffer *" << e->getEntryName() << "_" << curEntry << "_buf;\n";
    else if (e->paramIsMarshalled() == 1) {
      defs << "    CMsgBuffer *" << e->getEntryName() << "_" << curEntry << "_buf;\n";
      defs << "    CkMarshallMsg *" << e->getEntryName() << "_" << curEntry << "_msg;\n";
    }
    else {
      for (list<CStateVar*>::iterator it = e->stateVars.begin(); it != e->stateVars.end();
           ++it) {
        sv = *it;
        defs << "    CMsgBuffer *" << sv->name << "_buf;\n"
             << "    " << sv->type << " " << sv->name << ";\n";
      }
    }
    curEntry++;
    el = el->next;
  }

  bool singleEntry = curEntry == 1;

  defs << "\n";
  if (!singleEntry) defs << "    std::set<CMsgBuffer*> found;\n";
  el = elist;
  curEntry = 0;

  while (el != NULL) {
     e = el->entry;

     defs << "    ";
     generateEntryName(defs, e, curEntry);
     if (!singleEntry)
       defs << " = __cDep->getMessage(" << e->entryPtr->entryNum;
     else
       defs << " = __cDep->getMessageSingle(" << e->entryPtr->entryNum;

     if (e->intExpr)
       defs << ", " << e->intExpr;
     defs << ((!singleEntry) ? ", found" : "") << ");\n";

     if (!singleEntry) {
       defs << "    found.insert(";
       generateEntryName(defs, e, curEntry);
       defs << ");\n\n";
     }

    el = el->next;
    curEntry++;
  }

  defs << "\n";
  defs << "    if (";
  el = elist;
  curEntry = 0;

  while (el != NULL)  {
     e = el->entry;
     if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
       defs << "(" << e->getEntryName() << "_" << curEntry << "_buf != 0)";
     }
     else {
       sv = *(e->stateVars.begin());
       defs << "(" << sv->name << "_buf != 0)";
     }
     el = el->next;
     curEntry++;
     if (el != NULL)
        defs << "&&";
  }
  defs << ") {\n";

#if CMK_BIGSIM_CHARM
  // for tracing
  //TODO: instead of this, add a length field to EntryList
  int elen = 0;
  for(el=elist; el!=NULL; el=elist->next) elen++;
 
  defs << "         void * logs1["<< elen << "]; \n";
  defs << "         void * logs2["<< elen + 1<< "]; \n";
  int localnum = 0;

  curEntry = 0;
  for(el=elist; el!=NULL; el=elist->next) {
    e = el->entry;
       if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
         defs << "       logs1[" << localnum << "] = " << e->getEntryName() << "_" << curEntry << "_buf->bgLog1; \n";
         defs << "       logs2[" << localnum << "] = " << e->getEntryName() << "_" << curEntry << "_buf->bgLog2; \n";
	localnum++;
      }
      else{
	defs << "       logs1[" << localnum << "] = " << sv->name<< "_buf->bgLog1; \n";
	defs << "       logs2[" << localnum << "] = " << sv->name << "_buf->bgLog2; \n";
	localnum++;
      }
    curEntry++;
  }
      
  defs << "       logs2[" << localnum << "] = " << "_bgParentLog; \n";
  generateEventBracket(defs,SWHEN);
  defs << "       _TRACE_BG_FORWARD_DEPS(logs1,logs2,"<< localnum << ",_bgParentLog);\n";
#endif

  el = elist;
  curEntry = 0;

  while (el != NULL) {
     e = el->entry;
     if (e->param->isVoid() == 1) {
       defs <<"       CkFreeSysMsg((void *) "<< e->getEntryName() << "_" << curEntry <<"_buf->msg);\n";
       defs << "       __cDep->removeMessage(" << e->getEntryName() << "_" << curEntry <<
              "_buf);\n";
       defs << "      delete " << e->getEntryName() << "_" << curEntry << "_buf;\n";
     }
     else if (e->paramIsMarshalled() == 1) {
       defs << "       " << e->getEntryName() << "_" << curEntry << "_msg = (CkMarshallMsg *)"
            << e->getEntryName() << "_" << curEntry << "_buf->msg;\n";
       defs << "       char *"<<e->getEntryName() << "_" << curEntry <<"_impl_buf=((CkMarshallMsg *)"
            << e->getEntryName() << "_" << curEntry << "_msg)->msgBuf;\n";
       defs <<"       PUP::fromMem " << e->getEntryName() << "_" << curEntry <<"_implP("
            << e->getEntryName() << "_" << curEntry <<"_impl_buf);\n";

        for (list<CStateVar*>::iterator it = e->stateVars.begin(); it != e->stateVars.end();
             ++it) {
        CStateVar *sv = *it;
           if (sv->arrayLength != NULL)
             defs << "       int impl_off_"<<sv->name
                  << "; "<<e->getEntryName() << "_" << curEntry <<"_implP|impl_off_"
                  << sv->name<<";\n";
           else
             defs << "       "<<sv->type<<" "<<sv->name
                  << "; " <<e->getEntryName() << "_" << curEntry <<"_implP|"
                  << sv->name<<";\n";
	}
        defs << "       " <<e->getEntryName() << "_" << curEntry << "_impl_buf+=CK_ALIGN("
             << e->getEntryName() << "_" << curEntry << "_implP.size(),16);\n";
        for (list<CStateVar*>::iterator it = e->stateVars.begin(); it != e->stateVars.end();
             ++it) {
          CStateVar *sv = *it;
           if (sv->arrayLength != NULL)
              defs << "       "<<sv->type<< " *" <<sv->name <<"=(" <<sv->type
                   <<" *)(" << e->getEntryName() << "_" << curEntry << "_impl_buf+" <<"impl_off_"
                   << sv->name <<");\n";
        }
        defs << "       __cDep->removeMessage(" << e->getEntryName() << "_" << curEntry << "_buf);\n";
        defs << "       delete " << e->getEntryName() << "_" << curEntry << "_buf;\n";
     }
     else {  // There was a message as the only parameter
        sv = *e->stateVars.begin();
        defs << "       " << sv->name << " = (" <<
              sv->type << ") " <<
              sv->name << "_buf->msg;\n";
        defs << "       __cDep->removeMessage(" << sv->name <<
              "_buf);\n";
        defs << "       delete " << sv->name << "_buf;\n";
     }
     el = el->next;
     curEntry++;
  }

  // max(current,merge) --> current, then reset the mergepath
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "       " << label  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  defs << "       currentlyExecutingPath = " << label  << "_PathMergePoint; /* Critical Path Detection */ \n";
  defs << "       " << label  << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  if (speculativeState) {
    defs << "       __cDep->removeAllSpeculationIndex(" << speculativeState->name << "->speculationIndex);\n";
  }

  defs << "       ";

  if (constructs && !constructs->empty()) {
    generateCall(defs, *stateVarsChildren, constructs->front()->label);
  } else {
    generateCall(defs, *stateVarsChildren, label, "_end");
  }

  el = elist;
  curEntry = 0;
  while (el != NULL){
    e = el->entry;
    if (e->paramIsMarshalled() == 1) {
      defs << "       delete " << e->getEntryName() << "_" << curEntry << "_msg;\n";
    }
    el = el->next;
    curEntry++;
  }
  defs << "       return 0;\n";
  defs << "    } else {\n";

  int nRefs=0, nAny=0;
  el = elist;
  while (el != NULL) {
    e = el->entry;
    if(e->intExpr == 0)
      nAny++;
    else
      nRefs++;
    el = el->next;
  }
// keep these consts consistent with sdag.h in runtime

#define MAXARG 8
#define MAXANY 8
#define MAXREF 8

  if(stateVars->size() > MAXARG) {
    fprintf(stderr, "numStateVars more that %d, contact developers.\n",
		     MAXARG);
    exit(1);
  }
  if(nRefs > MAXREF) {
    fprintf(stderr, "numDepends more that %d, contact developers.\n",
		     MAXREF);
    exit(1);
  }
  if(nAny > MAXANY) {
    fprintf(stderr, "numDepends more that %d, contact developers.\n",
		     MAXANY);
    exit(1);
  }
  defs << "       CWhenTrigger *tr;\n";
  defs << "       tr = new CWhenTrigger(" << nodeNum << ", " <<
    (int)(stateVars->size()) << ", " << nRefs << ", " << nAny << ");\n";
  int iArgs=0;
 
//  defs << "       int impl_off=0;\n";
  int hasArray = 0;
  int numParamsNeedingMarshalling = 0;
  int paramIndex =0;
  for (list<CStateVar*>::iterator iter = stateVars->begin();
       iter != stateVars->end();
       ++iter) {
    CStateVar *sv = *iter;
    if (sv->isVoid == 1) {
       defs << "       if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
       defs << "       tr->args[" << iArgs++ << "] = NULL;\n";
    }
    else {
      if (sv->isMsg == 1 && !sv->isCounter && !sv->isSpeculator && !sv->isBgParentLog) {
         defs << "       if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
         defs << "       tr->args[" << iArgs++ << "] = new TransportableMsg(" << sv->name << ");\n";
      } else if (sv->isMsg == 1 && sv->isBgParentLog) {
         defs << "       if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
         defs << "       tr->args[" << iArgs++ << "] = new TransportableBigSimLog(" << sv->name << ");\n";
      } else if (sv->isMsg == 1 && (sv->isCounter || sv->isSpeculator)) {
         defs << "       if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
         defs << "       tr->args[" << iArgs++ << "] = " << sv->name << ";\n";
      } else {
         numParamsNeedingMarshalling++;
         if (numParamsNeedingMarshalling == 1) {
           defs << "       int impl_off=0;\n";
           paramIndex = iArgs;
           iArgs++;
         }
      }
      if (sv->arrayLength !=NULL) {
         hasArray++;
         if (hasArray == 1)
           defs<< "       int impl_arrstart=0;\n";
         defs <<"       int impl_off_"<<sv->name<<", impl_cnt_"<<sv->name<<";\n";
         defs <<"       impl_off_"<<sv->name<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type<<"));\n";
         defs <<"       impl_off+=(impl_cnt_"<<sv->name<<"=sizeof("<<sv->type<<")*("<<sv->arrayLength<<"));\n";
      }
    }
  }
  if (numParamsNeedingMarshalling > 0) {
     defs << "       { \n";
     defs << "         PUP::sizer implP;\n";
     for (list<CStateVar*>::iterator iter = stateVars->begin();
          iter != stateVars->end();
          ++iter) {
       CStateVar *sv = *iter;
       if (sv->arrayLength !=NULL)
         defs << "         implP|impl_off_" <<sv->name <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid !=1)) 
         defs << "         implP|" <<sv->name <<";\n";
     }
     if (hasArray > 0) {
        defs <<"         impl_arrstart=CK_ALIGN(implP.size(),16);\n";
        defs <<"         impl_off+=impl_arrstart;\n";
     }
     else {
        defs << "         impl_off+=implP.size();\n";
     }
     defs << "       }\n";
     defs << "       CkMarshallMsg *impl_msg;\n";
     defs << "       impl_msg = CkAllocateMarshallMsg(impl_off,NULL);\n";
     defs << "       {\n";
     defs << "         PUP::toMem implP((void *)impl_msg->msgBuf);\n";
     for (list<CStateVar*>::iterator iter = stateVars->begin();
          iter != stateVars->end();
          ++iter) {
       CStateVar *sv = *iter;
       if (sv->arrayLength !=NULL)
          defs << "         implP|impl_off_" <<sv->name <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid != 1))  
          defs << "         implP|" <<sv->name <<";\n";
     }
     defs << "       }\n";
     if (hasArray > 0) {
        defs <<"       char *impl_buf=impl_msg->msgBuf+impl_arrstart;\n";
        for (list<CStateVar*>::iterator iter = stateVars->begin();
             iter != stateVars->end();
             ++iter) {
          CStateVar *sv = *iter;
          if (sv->arrayLength !=NULL)
            defs << "       memcpy(impl_buf+impl_off_" << sv->name <<
              "," << sv->name << ",impl_cnt_" << sv->name << ");\n";
        }  
     }
  defs << "       if (tr->args[" << iArgs << "]) delete tr->args[" << iArgs << "];\n";
  defs << "       tr->args[" <<paramIndex <<"] = new TransportableMsg(impl_msg);\n";
  }
  int iRef=0, iAny=0;

  el = elist;
  while (el != NULL) {
    e = el->entry;
    if(e->intExpr == 0) {
      defs << "       tr->anyEntries[" << iAny++ << "] = " <<
            e->entryPtr->entryNum << ";\n";
    } else {
      defs << "       tr->entries[" << iRef << "] = " <<
            e->entryPtr->entryNum << ";\n";
      defs << "       tr->refnums[" << iRef++ << "] = " <<
            e->intExpr << ";\n";
    }
    el = el->next;
  }

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  // max(current,merge) --> current
  defs << "       " << label  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  defs << "       currentlyExecutingPath = " << label  << "_PathMergePoint; /* Critical Path Detection */ \n";
#endif

  defs << "       __cDep->Register(tr);\n";
  defs << "       return tr;\n";
  defs << "    }\n";

  endMethod(defs);

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs, SWHEN_END);
#endif
  defs << "    ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
  
  el = elist;
  while (el) {
    e = el->entry;
    if (e->param->isMessage() == 1) {
      sv = *e->stateVars.begin();
      defs << "    CmiFree(UsrToEnv(" << sv->name << "));\n";
    }

    el = el->next;
  }

  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void SdagConstruct::generateWhile(XStr& decls, XStr& defs, Entry* entry)
{
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

void SdagConstruct::generateFor(XStr& decls, XStr& defs, Entry* entry)
{
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());

  generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif

  unravelClosures(defs);

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
  endMethod(defs);

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");

  generateSignatureNew(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif
  unravelClosures(defs);

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
  endMethod(defs);
}

void SdagConstruct::unravelClosures(XStr& defs) {
  int cur = 0;
  // traverse all the state variables bring them into scope
  for (list<EncapState*>::iterator iter = encapState.begin(); iter != encapState.end(); ++iter, ++cur) {
    EncapState& state = **iter;

    defs << "  // begin encap: ";
    state.name ? (defs << *state.name) : (defs << "gen" << cur);
    defs << "\n";
    int i = 0;
    for (list<CStateVar*>::iterator iter2 = state.vars.begin(); iter2 != state.vars.end(); ++iter2, ++i) {
      CStateVar& var = **iter2;
      defs << "  " << var.type << (var.arrayLength ? "*" : "") << "& " << var.name << " = ";
      state.name ? (defs << *state.name) : (defs << "gen" << cur);
      defs << "->" << "getP" << i << "();\n";
    }
  }
}

void SdagConstruct::generateIf(XStr& decls, XStr& defs, Entry* entry)
{
  strcpy(nameStr,label->charstar());
  generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);

#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs, SIF);
#endif

  unravelClosures(defs);

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

void SdagConstruct::generateElse(XStr& decls, XStr& defs, Entry* entry)
{
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

void SdagConstruct::generateForall(XStr& decls, XStr& defs, Entry* entry)
{
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    int __first = (" << con2->text <<
        "), __last = (" << con3->text << 
        "), __stride = (" << con4->text << ");\n";
  defs << "    if (__first > __last) {\n";
  defs << "      int __tmp=__first; __first=__last; __last=__tmp;\n";
  defs << "      __stride = -__stride;\n";
  defs << "    }\n";
  defs << "    CCounter *" << counter <<
        " = new CCounter(__first,__last,__stride);\n"; 
  defs << "    for(int " << con1->text <<
        "=__first;" << con1->text <<
        "<=__last;" << con1->text << "+=__stride) {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label);
  defs << "    }\n";
  endMethod(defs);

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
  defs << "    " << counter << "->decrement(); /* DECREMENT 1 */ \n";
  defs << "    if (" << counter << "->isDone()) {\n";
  defs << "      delete " << counter << ";\n";
  defs << "      ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);
}

void SdagConstruct::generateOlist(XStr& decls, XStr& defs, Entry* entry)
{
  SdagConstruct *cn;
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    CCounter *" << counter << "= new CCounter(" <<
    (int)constructs->size() << ");\n";
  for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
       ++it) {
    defs << "    ";
    generateCall(defs, *stateVarsChildren, (*it)->label);
  }
  endMethod(defs);

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
#if CMK_BIGSIM_CHARM
  defs << "  CkVec<void*> " <<label << "_bgLogList;\n";
#endif

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
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

  defs << "      delete " << counter << ";\n";

#if CMK_BIGSIM_CHARM
  generateListEventBracket(defs, SOLIST_END);
  defs << "       "<< label <<"_bgLogList.length()=0;\n";
#endif

  defs << "      ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);
}

void SdagConstruct::generateOverlap(XStr& decls, XStr& defs, Entry* entry)
{
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs, SOVERLAP);
#endif
  defs << "    ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label);
  endMethod(defs);

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs, SOVERLAP_END);
#endif
  defs << "    ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void SdagConstruct::generateCaseList(XStr& decls, XStr& defs, Entry* entry) {
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    CSpeculator* " << counter << " = new CSpeculator(__dep->getAndIncrementSpeculationIndex());\n";
  defs << "    CWhenTrigger* tr = 0;\n";
  for (list<SdagConstruct*>::iterator it = constructs->begin(); it != constructs->end();
       ++it) {
    defs << "    tr = ";
    generateCall(defs, *stateVarsChildren, (*it)->label);
    defs << "    if (!tr) return;\n";
    defs << "    else tr->speculationIndex = " << counter << "->speculationIndex;\n";
  }
  endMethod(defs);

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);

  defs << "    delete " << counter << ";\n";
  defs << "    ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
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

  generateSignatureNew(decls, defs, entry, false, "void", con1->text, false, encapState);

#if CMK_BIGSIM_CHARM
  generateEndSeq(defs);
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor())
    generateTraceEndCall(defs);

  defs << "    if (!__dep.get())\n"
       << "        _sdag_init();\n";
  defs << "    ";
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
  endMethod(defs);
}

void AtomicConstruct::generateCodeNew(XStr& decls, XStr& defs, Entry* entry) {
  generateSignatureNew(decls, defs, entry, false, "void", label, false, encapState);

  unravelClosures(defs);

  defs << "  // begin serial block\n";
  defs << "  " << text << "\n";
  defs << "  // end serial block\n";
  defs << "  ";
  generateCallNew(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void AtomicConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  generateCodeNew(decls, defs, entry);
  return;

  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);

#if CMK_BIGSIM_CHARM
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateBeginExec(defs, nameStr);
#endif
  generateTraceBeginCall(defs);

  defs << "    " << text << "\n";

  generateTraceEndCall(defs);
#if CMK_BIGSIM_CHARM
  generateEndExec(defs);
#endif

  defs << "    ";
  generateCall(defs, *stateVars, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void generateSignature(XStr& str,
                       const XStr* name, const char* suffix,
                       list<CStateVar*>* params)
{

}
void generateSignature(XStr& decls, XStr& defs,
                       const Entry* entry, bool declareStatic, const char* returnType,
                       const XStr* name, bool isEnd,
                       list<CStateVar*>* params)
{
  generateSignature(decls, defs, entry->getContainer(), declareStatic, returnType,
                    name, isEnd, params);
}
void generateSignature(XStr& decls, XStr& defs,
                       const Chare* chare, bool declareStatic, const char* returnType,
                       const XStr* name, bool isEnd,
                       list<CStateVar*>* params)
{
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
void endMethod(XStr& op)
{
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
        op << "reinterpret_cast<" << *state->type << "*>(buf" << offset << "->packable)";
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

void SdagConstruct::generateCall(XStr& op, list<CStateVar*>& alist,
                                 const XStr* name, const char* nameSuffix) {
  op << name << (nameSuffix ? nameSuffix : "") << "(";

  CStateVar *sv;
  int isVoid;
  int count;
  count = 0;
  for (list<CStateVar*>::iterator iter = alist.begin(); iter != alist.end(); ++iter) {
    CStateVar *sv = *iter;
    isVoid = sv->isVoid;
    if ((count != 0) && (isVoid != 1))
      op << ", ";
    if (sv->name != 0) 
      op << sv->name;
    if (sv->isVoid != 1)
      count++;
  }

  op << ");\n";
}

// boe = 1, if the next call is to begin construct
// boe = 0, if the next call is to end a contruct
void SdagConstruct::setNext(SdagConstruct *n, int boe)
{
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
	  if ((*it)->type == SCONNECT)
	    continue;

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
    case SFORWARD:
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

void SdagConstruct::generateTrace()
{
  for_each(constructs->begin(), constructs->end(), mem_fun(&SdagConstruct::generateTrace));
  if (con1) con1->generateTrace();
  if (con2) con2->generateTrace();
  if (con3) con3->generateTrace();
}

void AtomicConstruct::generateTrace()
{
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

void SdagConstruct::generateTraceBeginCall(XStr& op)          // for trace
{
  if(traceName)
    op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (" << "_sdag_idx_" << traceName << "()), CkMyPe(), 0, NULL); \n";
}

void SdagConstruct::generateDummyBeginExecute(XStr& op)
{
    op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _dummyEP, CkMyPe(), 0, NULL); \n";
}

void SdagConstruct::generateTraceEndCall(XStr& op)          // for trace
{
  op << "    " << "_TRACE_END_EXECUTE(); \n";
}

void SdagConstruct::generateBeginExec(XStr& op, const char *name){
  op << "     " << "_TRACE_BG_BEGIN_EXECUTE_NOMSG(\""<<name<<"\",&_bgParentLog,1);\n"; 
}

void SdagConstruct::generateEndExec(XStr& op){
  op << "     " << "_TRACE_BG_END_EXECUTE(0);\n";
}

void SdagConstruct::generateBeginTime(XStr& op)
{
  //Record begin time for tracing
  op << "    double __begintime = CkVTimer(); \n";
}

void SdagConstruct::generateTlineEndCall(XStr& op)
{
  //Trace this event
  op <<"    _TRACE_BG_TLINE_END(&_bgParentLog);\n";
}

void SdagConstruct::generateEndSeq(XStr& op)
{
  op <<  "    void* _bgParentLog = NULL;\n";
  op <<  "    CkElapse(0.01e-6);\n";
  //op<<  "    BgElapse(1e-6);\n";
  generateTlineEndCall(op);
  generateTraceEndCall(op);
  generateEndExec(op);
}

void SdagConstruct::generateEventBracket(XStr& op, int eventType)
{
  (void) eventType;
  //Trace this event
  op << "     _TRACE_BG_USER_EVENT_BRACKET(\"" << nameStr
     << "\", __begintime, CkVTimer(),&_bgParentLog); \n";
}

void SdagConstruct::generateListEventBracket(XStr& op, int eventType)
{
  (void) eventType;
  op << "    _TRACE_BGLIST_USER_EVENT_BRACKET(\"" << nameStr
     << "\",__begintime,CkVTimer(),&_bgParentLog, " << label
     << "_bgLogList);\n";
}

void SdagConstruct::generateRegisterEp(XStr& defs)
{
  if (traceName) {
    defs << "    (void)_sdag_idx_" << traceName << "();\n";
  }

  for (list<SdagConstruct*>::iterator iter = constructs->begin(); iter != constructs->end(); ++iter)
    (*iter)->generateRegisterEp(defs);
  if (con1) con1->generateRegisterEp(defs);
  if (con2) con2->generateRegisterEp(defs);
  if (con3) con3->generateRegisterEp(defs);
}

void SdagConstruct::generateTraceEp(XStr& decls, XStr& defs, Chare* chare)
{
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


void RemoveSdagComments(char *str)
{
  char *ptr = str;
  while ((ptr = strstr(ptr, "//"))) {
    char *lend = strstr(ptr, "\n");
    if (lend==NULL) break;
    while (ptr != lend) *ptr++=' ';
  }
}

}
