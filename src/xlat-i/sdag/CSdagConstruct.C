#include <string.h>
#include <stdlib.h>
#include "sdag-globals.h"
#include "xi-symbol.h"
//#include "CParsedFile.h"
#include "EToken.h"
#include "CStateVar.h"

namespace xi {

SdagConstruct *buildAtomic(const char* code,
			   SdagConstruct *pub_list,
			   const char *trace_name)
{
  char *tmp = strdup(code);
  RemoveSdagComments(tmp);
  SdagConstruct *ret = new SdagConstruct(SATOMIC, new XStr(tmp), pub_list, 0,0,0,0, 0 );
  free(tmp);

  if (trace_name)
  {
    tmp = strdup(trace_name);
    tmp[strlen(tmp)-1]=0;
    ret->traceName = new XStr(tmp+1);
    free(tmp);
  }

  return ret;
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
    case SINT_EXPR:
    case SIDENT: 
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->numberNodes();
    }
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
    case SINT_EXPR:
    case SIDENT:
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for(cn=(SdagConstruct *)(constructs->begin()); !constructs->end(); cn=(SdagConstruct *)(constructs->next())) {
      cn->labelNodes();
    }
  }
}

void EntryList::generateEntryList(TList<CEntry*>& CEntrylist, SdagConstruct *thisWhen)
{
   EntryList *el;
   el = this;
   while (el != NULL)
   {
     el->entry->generateEntryList(CEntrylist, thisWhen);
     el = el->next;
   }
}

void Entry::generateEntryList(TList<CEntry*>& CEntrylist, SdagConstruct *thisWhen)
{
   // case SENTRY:
   CEntry *entry;
   bool found = false;
   
   for(entry=CEntrylist.begin(); !CEntrylist.end(); entry=CEntrylist.next()) {
     if(*(entry->entry) == (const char *)name) 
     {
        ParamList *epl;
	epl = entry->paramlist;
        ParamList *pl;
        pl = param;
        found = false;
	if ((entry->paramlist->isVoid() == 1) && (pl->isVoid() == 1)) {
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
          int whenFound = 0;
          TList<SdagConstruct*> *tmpList = &(entry->whenList);
          SdagConstruct *tmpNode;
          for(tmpNode = tmpList->begin(); !tmpList->end(); tmpNode = tmpList->next()) {
            if(tmpNode->nodeNum == thisWhen->nodeNum)
               whenFound = 1;
          }
          if(!whenFound)
            entry->whenList.append(thisWhen);
          entryPtr = entry;
          if(intExpr != 0)
            entry->refNumNeeded = 1; 
	 } 
     }
   }
   if(!found) {
     CEntry *newEntry;
     newEntry = new CEntry(new XStr(name), param, estateVars, paramIsMarshalled() );
     CEntrylist.append(newEntry);
     entryPtr = newEntry;
     newEntry->whenList.append(thisWhen);
     if(intExpr != 0)
       newEntry->refNumNeeded = 1; 
   }
      //break;
}

void SdagConstruct::generateEntryList(TList<CEntry*>& CEntrylist, SdagConstruct *thisWhen)
{
  SdagConstruct *cn;
  switch(type) {
    case SWHEN:
      elist->generateEntryList(CEntrylist, this);  /* con1 is the WHEN's ELIST */
      break;
    case SIF:
	/* con2 is the ELSE corresponding to this IF */
      if(con2!=0) con2->generateEntryList(CEntrylist, thisWhen); 
      break;
  }
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateEntryList(CEntrylist,thisWhen);
    }
  }
}
 
void SdagConstruct::generateConnectEntries(XStr& op){
   op << "  void " <<connectEntry->charstar() <<'(';
   ParamList *pl = param;
   XStr msgParams;
   if (pl->isVoid() == 1) {
     op << "void) {\n"; 
   }
   else if (pl->isMessage() == 1){
     op << pl->getBaseName() <<" *" <<pl->getGivenName() <<") {\n";
   }
   else {
    op << "CkMarshallMsg *" /*<< connectEntry->charstar()*/ <<"_msg) {\n";
    msgParams <<"   char *impl_buf= _msg->msgBuf;\n";
    param->beginUnmarshall(msgParams);
   }
   op << msgParams.charstar() <<"\n"; 
   op << "  " <<text->charstar() <<"\n";

   op << "  }\n";
   
}

void SdagConstruct::generateConnectEntryList(TList<SdagConstruct*>& ConnectEList) {
  if (type == SCONNECT)
     ConnectEList.append(this);
  if (constructs != 0) {
    SdagConstruct *cn;
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateConnectEntryList(ConnectEList);
    }
  }

}

void SdagConstruct::propagateState(int uniqueVarNum)
{ 
  CStateVar *sv; 
  /*if(type != SSDAGENTRY) {
    fprintf(stderr, "use of non-entry as the outermost construct..\n");
    exit(1);
  }*/
  stateVars = new TList<CStateVar*>();
  ParamList *pl = param;
  if (pl->isVoid() == 1) {
     sv = new CStateVar(1, NULL, 0, NULL, 0, NULL, 0);
     stateVars->append(sv);
  }
  else {
    while (pl != NULL) {
      stateVars->append(new CStateVar(pl));
      pl = pl->next;
    }
  }

#if CMK_BIGSIM_CHARM
  // adding _bgParentLog as the last extra parameter for tracing
  stateVarsChildren = new TList<CStateVar*>();

  for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next())
    stateVarsChildren->append(sv);
  sv = new CStateVar(0, "void *", 0,"_bgParentLog", 0, NULL, 1);  
  stateVarsChildren->append(sv);
#else
  stateVarsChildren = stateVars; 
#endif

  SdagConstruct *cn;
  TList<CStateVar*> *whensEntryMethodStateVars; 
  whensEntryMethodStateVars = new TList<CStateVar*>();
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
     cn->propagateState(*stateVarsChildren, *whensEntryMethodStateVars , *publishesList, uniqueVarNum);
  }
}


void SdagConstruct::propagateState(TList<CStateVar*>& list, TList<CStateVar*>& wlist, TList<SdagConstruct*>& publist, int uniqueVarNum)
{
  CStateVar *sv;
  TList<CStateVar*> *whensEntryMethodStateVars; 
  stateVars = new TList<CStateVar*>();
  switch(type) {
    case SFORALL:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      sv = new CStateVar(0,"int", 0, con1->text->charstar(), 0,NULL, 0);
      stateVarsChildren->append(sv);
      {
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "CCounter *", 0, txt, 0, NULL, 1);
        stateVarsChildren->append(sv);
      }
      break;
    case SWHEN:
      whensEntryMethodStateVars = new TList<CStateVar*>();
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
     
      {  
        EntryList *el;
	el = elist;
        ParamList *pl;
	while (el != NULL) {
          pl = el->entry->param;
	  el->entry->stateVars = new TList<CStateVar*>();
          if (pl->isVoid()) {
            sv = new CStateVar(1, NULL, 0, NULL, 0, NULL, 0);
            //stateVars->append(sv);
              stateVarsChildren->append(sv);
              whensEntryMethodStateVars->append(sv); 
 	      el->entry->estateVars.append(sv);
 	      el->entry->stateVars->append(sv);
          }
          else {
            while(pl != NULL) {
              sv = new CStateVar(pl);
              stateVarsChildren->append(sv);
              whensEntryMethodStateVars->append(sv); 
 	      el->entry->estateVars.append(sv);
 	      el->entry->stateVars->append(sv);

              pl = pl->next;
	    }
	  }
	  el = el->next;

	}
      }
      break;
    case SIF:
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      if(con2 != 0) con2->propagateState(list, wlist,publist, uniqueVarNum);
      break;
    case SOLIST:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      {
        char txt[128];
        sprintf(txt, "_co%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(0, "CCounter *", 0, txt, 0, NULL, 1);
        stateVarsChildren->append(sv);
      }
      break;
    case SFOR:
    case SWHILE:
    case SELSE:
    case SSLIST:
    case SOVERLAP:
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      break;
    case SATOMIC:
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      if (con1 != 0) {
        publist.append(con1);
        /*SdagConstruct *sc;
        SdagConstruct *sc1;
        for(sc =publist.begin(); !publist.end(); sc=publist.next()) {
           for(sc1=sc->constructs->begin(); !sc->constructs->end(); sc1 = sc->constructs->next())
           printf("Publist = %s\n", sc1->text->charstar());
	}*/
      }
      break;
    case SFORWARD:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) { 
        stateVars->append(sv);
      }
      for(sv=wlist.begin(); !wlist.end(); sv=wlist.next()) { 
        stateVarsChildren->append(sv);
      }

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
  SdagConstruct *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      if (type == SWHEN)
         cn->propagateState(*stateVarsChildren, *whensEntryMethodStateVars, publist,  uniqueVarNum);
      else
         cn->propagateState(*stateVarsChildren, wlist, publist,  uniqueVarNum);
    }
 } 
}

void SdagConstruct::generateCode(XStr& op, Entry *entry)
{
  switch(type) {
    case SSDAGENTRY:
      generateSdagEntry(op, entry);
      break;
    case SSLIST:
      generateSlist(op);
      break;
    case SOLIST:
      generateOlist(op);
      break;
    case SFORALL:
      generateForall(op);
      break;
    case SATOMIC:
      generateAtomic(op);
      break;
    case SIF:
      generateIf(op);
      if(con2 != 0)
        con2->generateCode(op, entry);
      break;
    case SELSE:
      generateElse(op);
      break;
    case SWHILE:
      generateWhile(op);
      break;
    case SFOR:
      generateFor(op);
      break;
    case SOVERLAP:
      generateOverlap(op);
      break;
    case SWHEN:
      generateWhen(op);
      break;
    case SFORWARD:
      generateForward(op);
      break;
    case SCONNECT:
      generateConnect(op);
      break;
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateCode(op, entry);
    }
  }
}

void SdagConstruct::generateConnect(XStr& op) {
  op << "  void " << label->charstar() << "() {\n";
  op << "    int index;\n";
  if ((param->isVoid() == 0) && (param->isMessage() == 0)) {
     op << "    CkMarshallMsg *x;\n";  
     op << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(x);\n";  //replace
     op << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace 
  }
  else if (param->isVoid() == 1) {
     op << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(void);\n";  //replace
     op << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace 
  }
  else {
     op << "    " << param->getBaseName() <<" *x;\n";  // replace
     op << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(x);\n";  //replace
     op << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace 
  }
  op << "    myPublish->get_" <<connectEntry->charstar() <<"(cb);\n";  //replace - myPublish

  op << "  }\n\n";  
}

void SdagConstruct::generateForward(XStr& op) {
  SdagConstruct *cn;
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  for (cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    op << "    { ";
    generateCall(op, *stateVarsChildren, cn->text->charstar());
    op<<" }\n";
  }
  generateCall(op, *stateVarsChildren, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "  }\n\n";
}


void SdagConstruct::generateWhen(XStr& op)
{
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  op << "  int " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
#endif

  CStateVar *sv;

  Entry *e;
  EntryList *el;
  el = elist;
  while (el != NULL){
    e = el->entry;
    if (e->param->isVoid() == 1)
        op << "    CMsgBuffer *"<<e->getEntryName()<<"_buf;\n";
    else if (e->paramIsMarshalled() == 1) {

        op << "    CMsgBuffer *"<<e->getEntryName()<<"_buf;\n";
        op << "    CkMarshallMsg *" <<
                        e->getEntryName() << "_msg;\n";
    }
    else {
        for(sv=e->stateVars->begin(); !e->stateVars->end(); e->stateVars->next()) {
          op << "    CMsgBuffer *"<<sv->name->charstar()<<"_buf;\n";
          op << "    " << sv->type->charstar() << " " <<
                          sv->name->charstar() << ";\n";
        } 
    }
    el = el->next;
  }

  op << "\n";
  el = elist;
  while (el != NULL) {
     e = el->entry;
     if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() == 1)) {
        if((e->intExpr == 0) || (e->param->isVoid() == 1)) {   // DOUBLE CHECK THIS LOGIC
           op << "    " << e->getEntryName(); 
           op << "_buf = __cDep->getMessage(" << e->entryPtr->entryNum << ");\n";
        }	    
        else {
           op << "    " << e->getEntryName() << 
                 "_buf = __cDep->getMessage(" << e->entryPtr->entryNum <<
                 ", " << e->intExpr << ");\n";
        }
     }
     else { // The parameter is a message
        sv = e->stateVars->begin();
        if(e->intExpr == 0) {
           op << "    " << sv->name->charstar(); 
           op << "_buf = __cDep->getMessage(" << e->entryPtr->entryNum << ");\n";
        }	    
        else {
           op << "    " << sv->name->charstar() << 
                 "_buf = __cDep->getMessage(" << e->entryPtr->entryNum <<
                 ", " << e->intExpr << ");\n";
        }
     }  
    el = el->next;
  }

  op << "\n";
  op << "    if (";
  el = elist;
  while (el != NULL)  {
     e = el->entry;
     if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
        op << "(" << e->getEntryName() << "_buf != 0)";
     }
     else {
        sv = e->stateVars->begin();
        op << "(" << sv->name->charstar() << "_buf != 0)";
     }
     el = el->next;
     if (el != NULL)
        op << "&&";
  }
  op << ") {\n";

#if CMK_BIGSIM_CHARM
  // for tracing
  //TODO: instead of this, add a length field to EntryList
  int elen=0;
  for(el=elist; el!=NULL; el=elist->next) elen++;
 
  op << "         void * logs1["<< elen << "]; \n";
  op << "         void * logs2["<< elen + 1<< "]; \n";
  int localnum = 0;
  for(el=elist; el!=NULL; el=elist->next) {
    e = el->entry;
       if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
	op << "       logs1[" << localnum << "] = " << /*el->con4->text->charstar() sv->type->charstar()*/e->getEntryName() << "_buf->bgLog1; \n";
	op << "       logs2[" << localnum << "] = " << /*el->con4->text->charstar() sv->type->charstar()*/e->getEntryName() << "_buf->bgLog2; \n";
	localnum++;
      }
      else{
	op << "       logs1[" << localnum << "] = " << /*el->con4->text->charstar()*/ sv->name->charstar()<< "_buf->bgLog1; \n";
	op << "       logs2[" << localnum << "] = " << /*el->con4->text->charstar()*/ sv->name->charstar() << "_buf->bgLog2; \n";
	localnum++;
      }
  }
      
  op << "       logs2[" << localnum << "] = " << "_bgParentLog; \n";
  generateEventBracket(op,SWHEN);
  op << "       _TRACE_BG_FORWARD_DEPS(logs1,logs2,"<< localnum << ",_bgParentLog);\n";
#endif

  el = elist;
  while (el != NULL) {
     e = el->entry;
     if (e->param->isVoid() == 1) {
        op <<"       CkFreeSysMsg((void *) "<<e->getEntryName() <<"_buf->msg);\n";
        op << "       __cDep->removeMessage(" << e->getEntryName() <<
              "_buf);\n";
        op << "      delete " << e->getEntryName() << "_buf;\n";
     }
     else if (e->paramIsMarshalled() == 1) {
        op << "       " << e->getEntryName() << "_msg = (CkMarshallMsg *)"  
               << e->getEntryName() << "_buf->msg;\n";
        op << "       char *"<<e->getEntryName() <<"_impl_buf=((CkMarshallMsg *)"
	   <<e->getEntryName() <<"_msg)->msgBuf;\n";
        op <<"       PUP::fromMem " <<e->getEntryName() <<"_implP(" 
	   <<e->getEntryName() <<"_impl_buf);\n";

        for(sv=e->stateVars->begin(); !e->stateVars->end(); sv=e->stateVars->next()) {
           if (sv->arrayLength != NULL)
              op <<"      int impl_off_"<<sv->name->charstar()
	         <<"; "<<e->getEntryName() <<"_implP|impl_off_"
		 <<sv->name->charstar()<<";\n";
           else
               op <<"       "<<sv->type->charstar()<<" "<<sv->name->charstar()
	       <<"; " <<e->getEntryName() <<"_implP|"
	       <<sv->name->charstar()<<";\n";
	}
        op << "       " <<e->getEntryName() <<"_impl_buf+=CK_ALIGN("
	   <<e->getEntryName() <<"_implP.size(),16);\n";
        for(sv=e->stateVars->begin(); !e->stateVars->end(); sv=e->stateVars->next()) {
           if (sv->arrayLength != NULL)
              op << "    "<<sv->type->charstar()<< " *" <<sv->name->charstar() <<"=(" <<sv->type->charstar()
		 <<" *)(" <<e->getEntryName() <<"_impl_buf+" <<"impl_off_"
		 <<sv->name->charstar()<<");\n";
        }
        op << "       __cDep->removeMessage(" << e->getEntryName() <<
              "_buf);\n";
        op << "       delete " << e->getEntryName() << "_buf;\n";
     }
     else {  // There was a message as the only parameter
        sv = e->stateVars->begin();
        op << "       " << sv->name->charstar() << " = (" << 
              sv->type->charstar() << ") " <<
              sv->name->charstar() << "_buf->msg;\n";
        op << "       __cDep->removeMessage(" << sv->name->charstar() <<
              "_buf);\n";
        op << "       delete " << sv->name->charstar() << "_buf;\n";
     }
     el = el->next;
  }

  // max(current,merge) --> current, then reset the mergepath
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  op << "       " << label->charstar()  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  op << "       currentlyExecutingPath = " << label->charstar()  << "_PathMergePoint; /* Critical Path Detection */ \n";
  op << "       " << label->charstar()  << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  op << "       ";

  if (constructs && !constructs->empty()) {
    generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  } else {
    generateCall(op, *stateVarsChildren, label->charstar(), "_end");
  }

  el = elist;
  while (el != NULL){
    e = el->entry;
    if (e->paramIsMarshalled() == 1) {
        op << "       delete " << e->getEntryName() << "_msg;\n";
    }
    el = el->next;
  }
  op << "       return 1;\n";
  op << "    } else {\n";

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

  if(stateVars->length() > MAXARG) {
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
  op << "       CWhenTrigger *tr;\n";
  op << "       tr = new CWhenTrigger(" << nodeNum << ", " <<
        stateVars->length() << ", " << nRefs << ", " << nAny << ");\n";
  int iArgs=0;
 
//  op << "       int impl_off=0;\n";
  int hasArray = 0;
  int numParamsNeedingMarshalling = 0;
  int paramIndex =0;
  for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
    if (sv->isVoid == 1) {
       // op <<"       tr->args[" <<iArgs++ <<"] = (size_t) CkAllocSysMsg();\n";
       op <<"       tr->args[" <<iArgs++ <<"] = (size_t)0xFF;\n";
    }
    else {
      if (sv->isMsg == 1) {
         op << "       tr->args["<<iArgs++ <<"] = (size_t) " <<sv->name->charstar()<<";\n";
      }
      else {
         numParamsNeedingMarshalling++;
         if (numParamsNeedingMarshalling == 1) {
           op << "       int impl_off=0;\n";
           paramIndex = iArgs;
           iArgs++;
         }
      }
      if (sv->arrayLength !=NULL) {
         hasArray++;
         if (hasArray == 1)
      	   op<< "       int impl_arrstart=0;\n";
         op <<"       int impl_off_"<<sv->name->charstar()<<", impl_cnt_"<<sv->name->charstar()<<";\n";
         op <<"       impl_off_"<<sv->name->charstar()<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type->charstar()<<"));\n";
         op <<"       impl_off+=(impl_cnt_"<<sv->name->charstar()<<"=sizeof("<<sv->type->charstar()<<")*("<<sv->arrayLength->charstar()<<"));\n";
      }
    }
  }
  if (numParamsNeedingMarshalling > 0) {
     op << "       { \n";
     op << "         PUP::sizer implP;\n";
     for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
       if (sv->arrayLength !=NULL)
         op << "         implP|impl_off_" <<sv->name->charstar() <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid !=1)) 
         op << "         implP|" <<sv->name->charstar() <<";\n";
     }
     if (hasArray > 0) {
        op <<"         impl_arrstart=CK_ALIGN(implP.size(),16);\n";
        op <<"         impl_off+=impl_arrstart;\n";
     }
     else {
        op << "         impl_off+=implP.size();\n";
     }
     op << "       }\n";
     op << "       CkMarshallMsg *impl_msg;\n";
     op << "       impl_msg = CkAllocateMarshallMsg(impl_off,NULL);\n";
     op << "       {\n";
     op << "         PUP::toMem implP((void *)impl_msg->msgBuf);\n";
     for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
       if (sv->arrayLength !=NULL)
          op << "         implP|impl_off_" <<sv->name->charstar() <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid != 1))  
          op << "         implP|" <<sv->name->charstar() <<";\n";
     }
     op << "       }\n";
     if (hasArray > 0) {
        op <<"       char *impl_buf=impl_msg->msgBuf+impl_arrstart;\n";
        for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
           if (sv->arrayLength !=NULL)
              op << "       memcpy(impl_buf+impl_off_"<<sv->name->charstar()<<
	                 ","<<sv->name->charstar()<<",impl_cnt_"<<sv->name->charstar()<<");\n";
        }  
     }
  op << "       tr->args[" <<paramIndex <<"] = (size_t) impl_msg;\n";
  }
  int iRef=0, iAny=0;

  el = elist;
  while (el != NULL) {
    e = el->entry;
    if(e->intExpr == 0) {
      op << "       tr->anyEntries[" << iAny++ << "] = " <<
            e->entryPtr->entryNum << ";\n";
    } else {
      op << "       tr->entries[" << iRef << "] = " << 
            e->entryPtr->entryNum << ";\n";
      op << "       tr->refnums[" << iRef++ << "] = " <<
            e->intExpr << ";\n";
    }
    el = el->next;
  }

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  // max(current,merge) --> current
  op << "       " << label->charstar()  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  op << "       currentlyExecutingPath = " << label->charstar()  << "_PathMergePoint; /* Critical Path Detection */ \n";
#endif

  op << "       __cDep->Register(tr);\n";
  op << "       return 0;\n";
  op << "    }\n";

  // end actual code
  op << "  }\n\n";

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");

  // end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SWHEN_END);
#endif
  // actual code here 
  op << "    ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  
  el = elist;
  while (el) {
    e = el->entry;
    if (e->param->isMessage() == 1) {
      sv = e->stateVars->begin();
      op << "    CmiFree(UsrToEnv(" << sv->name->charstar() << "));\n";
    }

    el = el->next;
  }

  // end actual code
  op << "  }\n\n";
}

void SdagConstruct::generateWhile(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    } else {\n";
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    } else {\n";
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
}

void SdagConstruct::generateFor(XStr& op)
{
  // inlined start function
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  // actual code here 
  generateBeginTime(op);
#endif
  op << "    " << con1->text->charstar() << ";\n";
  //Record only the beginning for FOR
#if CMK_BIGSIM_CHARM
  generateEventBracket(op,SFOR);
#endif
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    } else {\n";
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "    }\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
#endif
  // actual code here 
  op << con3->text->charstar() << ";\n";
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    } else {\n";
#if CMK_BIGSIM_CHARM
  generateEventBracket(op,SFOR_END);
#endif
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "    }\n";
  // end actual code
  op << "  }\n";
}

void SdagConstruct::generateIf(XStr& op)
{
  // inlined start function
  strcpy(nameStr,label->charstar());
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SIF);
#endif
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    } else {\n";
  op << "      ";
  if (con2 != 0) {
    generateCall(op, *stateVarsChildren, con2->label->charstar());
  } else {
    generateCall(op, *stateVarsChildren, label->charstar(), "_end");
  }
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  strcpy(nameStr,label->charstar());
  strcat(nameStr,"_end");
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SIF_END);
#endif
  // actual code here 
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  // end actual code
  op << "  }\n\n";
}

void SdagConstruct::generateElse(XStr& op)
{
  // inlined start function
  strcpy(nameStr,label->charstar());
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // trace
  generateBeginTime(op);
  generateEventBracket(op,SELSE);
  // actual code here 
  op << "    ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  // end actual code
  op << "  }\n\n";
  // inlined end function
  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SELSE_END);
#endif
  // actual code here 
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  // end actual code
  op << "  }\n\n";
}

void SdagConstruct::generateForall(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    int __first = (" << con2->text->charstar() <<
        "), __last = (" << con3->text->charstar() << 
        "), __stride = (" << con4->text->charstar() << ");\n";
  op << "    if (__first > __last) {\n";
  op << "      int __tmp=__first; __first=__last; __last=__tmp;\n";
  op << "      __stride = -__stride;\n";
  op << "    }\n";
  op << "    CCounter *" << counter->charstar() << 
        " = new CCounter(__first,__last,__stride);\n"; 
  op << "    for(int " << con1->text->charstar() << 
        "=__first;" << con1->text->charstar() <<
        "<=__last;" << con1->text->charstar() << "+=__stride) {\n";
  op << "      ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    " << counter->charstar() << "->decrement(); /* DECREMENT 1 */ \n";
  op << "    if (" << counter->charstar() << "->isDone()) {\n";
  op << "      delete " << counter->charstar() << ";\n";
  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  // end actual code
  op << "    }\n  }\n\n";
}

void SdagConstruct::generateOlist(XStr& op)
{
  SdagConstruct *cn;
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    CCounter *" << counter->charstar() << "= new CCounter(" <<
        constructs->length() << ");\n";
  for(cn=constructs->begin(); 
                     !constructs->end(); cn=constructs->next()) {
    op << "    ";
    generateCall(op, *stateVarsChildren, cn->label->charstar());
  }
  // end actual code
  op << "  }\n";

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
#if CMK_BIGSIM_CHARM
  op << "  CkVec<void*> " <<label->charstar() << "_bgLogList;\n";
#endif

  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  op << "    " <<label->charstar() << "_bgLogList.insertAtEnd(_bgParentLog);\n";
#endif
  // actual code here 
  //Accumulate all the bgParent pointers that the calling when_end functions give
  op << "    " << counter->charstar() << "->decrement();\n";
 
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
 op << "    olist_" << counter->charstar() << "_PathMergePoint.updateMax(currentlyExecutingPath);  /* Critical Path Detection FIXME: is the currently executing path the right thing for this? The duration ought to have been added somewhere. */ \n";
#endif

  op << "    if (" << counter->charstar() << "->isDone()) {\n";

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  op << "      currentlyExecutingPath = olist_" << counter->charstar() << "_PathMergePoint; /* Critical Path Detection */ \n";
  op << "      olist_" << counter->charstar() << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  op << "      delete " << counter->charstar() << ";\n";

#if CMK_BIGSIM_CHARM
  generateListEventBracket(op,SOLIST_END);
  op << "       "<< label->charstar() <<"_bgLogList.length()=0;\n";
#endif

  op << "      ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  // end actual code
  op << "    }\n";
  op << "  }\n";
}

void SdagConstruct::generateOverlap(XStr& op)
{
  // inlined start function
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SOVERLAP);
#endif
  // actual code here 
  op << "    ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  // end actual code
  op << "  }\n";
  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginTime(op);
  generateEventBracket(op,SOVERLAP_END);
#endif
  // actual code here 
  op << "    ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << ");\n";
  // end actual code
  op << "  }\n";
}

void SdagConstruct::generateSlist(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren); 
  op << ") {\n";
  // actual code here 
  op << "    ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  // end actual code
  op << "  }\n";
}

void SdagConstruct::generateSdagEntry(XStr& op, Entry *entry)
{
  // header file
  op << "public:\n";
  op << "  void " << con1->text->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  SdagConstruct *sc;
  SdagConstruct *sc1;
  for(sc =publishesList->begin(); !publishesList->end(); sc=publishesList->next()) {
     for(sc1=sc->constructs->begin(); !sc->constructs->end(); sc1 = sc->constructs->next())
        op << "    _connect_" << sc1->text->charstar() <<"();\n";
  }

#if CMK_BIGSIM_CHARM
  generateEndSeq(op);
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor()) generateTraceEndCall(op);

  // actual code here 
  op << "    ";
  generateCall(op, *stateVarsChildren, constructs->front()->label->charstar());

#if CMK_BIGSIM_CHARM
  generateTlineEndCall(op);
  generateBeginExec(op, "spaceholder");
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor()) generateDummyBeginExecute(op);

  // end actual code
  op << "  }\n\n";
  op << "private:\n";
  op << "  void " << con1->text->charstar() << "_end(";
#if CMK_BIGSIM_CHARM
  generatePrototype(op, *stateVarsChildren);
#else
  generatePrototype(op, *stateVars);
#endif
  op << ") {\n";
  op << "  }\n\n";
}

void SdagConstruct::generateAtomic(XStr& op)
{ 
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
#if CMK_BIGSIM_CHARM
  generateBeginExec(op, nameStr);
#endif
  generateTraceBeginCall(op);
  op << "    " << text->charstar() << "\n";
  // trace
  generateTraceEndCall(op);
#if CMK_BIGSIM_CHARM
  generateEndExec(op);
#endif
  op << "    ";
  generateCall(op, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  op << "  }\n\n";
}

void SdagConstruct::generatePrototype(XStr& op, ParamList *list)
{
   ParamList *pl = list;
   int count = 0;

   if (pl->isVoid() == 1) {
     op << "void"; 
   }
   else {
     while(pl != NULL) {
       if (count > 0)
          op <<", ";
       op << pl->param->getType();

       if (pl->isReference())
         op << "&";

       op << pl->getGivenName();

       pl = pl->next;
       count++;
     }
   }
}


void SdagConstruct::generatePrototype(XStr& op, TList<CStateVar*>& list)
{
  CStateVar *sv;
  int isVoid;
  int count;
  count = 0;
  for(sv=list.begin(); !list.end(); ) {
    isVoid = sv->isVoid;
    if ((count != 0) && (isVoid != 1))
       op << ", ";
    if (sv->isVoid != 1) {
      if (sv->type != 0) 
         op <<sv->type->charstar() <<" ";
      if (sv->byRef != 0)
         op <<" &";
      if (sv->arrayLength != NULL) 
        op <<"* ";
      if (sv->name != 0)
         op <<sv->name->charstar();
    }
    if (sv->isVoid != 1)
       count++;
    sv = list.next();
  }
}

void SdagConstruct::generateCall(XStr& op, TList<CStateVar*>& list,
                                 const char* name, const char* nameSuffix) {
  op << name << (nameSuffix ? nameSuffix : "") << "(";

  CStateVar *sv;
  int isVoid;
  int count;
  count = 0;
  for(sv=list.begin(); !list.end(); ) {
     isVoid = sv->isVoid;
     if ((count != 0) && (isVoid != 1))
        op << ", ";
     if (sv->name != 0) 
       op << sv->name->charstar();
    if (sv->isVoid != 1)
       count++;
    sv = list.next();
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
        SdagConstruct *cn=constructs->begin();
        if (cn==0) // empty slist
          return;

        for(SdagConstruct *nextNode=constructs->next(); nextNode != 0; nextNode = constructs->next()) {
	  if (nextNode->type == SCONNECT)
	    continue;

          cn->setNext(nextNode, 1);
          cn = nextNode;
        }
        cn->setNext(this, 0);
      }
      return;
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
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->setNext(n, boe);
    }
  }
}


// for trace

void SdagConstruct::generateTrace()
{
  char text[1024];
  switch(type) {
  case SATOMIC:
    if (traceName) {
      sprintf(text, "%s_%s", CParsedFile::className->charstar(), traceName->charstar());
      // remove blanks
      for (unsigned int i=0; i<strlen(text); i++)
        if (text[i]==' '||text[i]=='\t') text[i]='_';
    }
    else {
      sprintf(text, "%s%s", CParsedFile::className->charstar(), label->charstar());
    }
    traceName = new XStr(text);
    break;
  default:
    break;
  }

  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateTrace();
  }
  if (con1) con1->generateTrace();
  if (con2) con2->generateTrace();
  if (con3) con3->generateTrace();
}

void SdagConstruct::generateTraceBeginCall(XStr& op)          // for trace
{
  if(traceName)
    op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, " << "__idx_" << traceName->charstar() << ", CkMyPe(), 0, NULL); \n";
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
     << "\",__begintime,CkVTimer(),&_bgParentLog, " << label->charstar()
     << "_bgLogList);\n";
}

void SdagConstruct::generateRegisterEp(XStr& op)          // for trace
{
  if (traceName) {
    op << "    __idx_" << traceName->charstar()
       << " = CkRegisterEp(\"" << traceName->charstar()
       << "(void)\", NULL, 0, CkIndex_" << CParsedFile::className->charstar()
       << "::__idx, 0);\n";
  }

  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateRegisterEp(op);
  }
  if (con1) con1->generateRegisterEp(op);
  if (con2) con2->generateRegisterEp(op);
  if (con3) con3->generateRegisterEp(op);
}

void SdagConstruct::generateTraceEpDecl(XStr& op)          // for trace
{
  if (traceName) {
    op << "  static int __idx_" << traceName->charstar() << ";\n"; 
  }
  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateTraceEpDecl(op);
  }
  if (con1) con1->generateTraceEpDecl(op);
  if (con2) con2->generateTraceEpDecl(op);
  if (con3) con3->generateTraceEpDecl(op);
}


void SdagConstruct::generateTraceEpDef(XStr& op)          // for trace
{
  if (traceName) {
    op << "  int " << CParsedFile::className->charstar()
       << "::__idx_" << traceName->charstar() << "=0;\\\n";
  }
  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateTraceEpDef(op);
  }
  if (con1) con1->generateTraceEpDef(op);
  if (con2) con2->generateTraceEpDef(op);
  if (con3) con3->generateTraceEpDef(op);
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
