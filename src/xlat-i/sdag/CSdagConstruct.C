#include <string.h>
#include <stdlib.h>
#include "sdag-globals.h"
#include "xi-symbol.h"
#include "CParsedFile.h"
#include "EToken.h"
#include "CStateVar.h"

namespace xi {

SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1)
{
  con1 = 0;  con2 = 0; con3 = 0; con4 = 0;
  type = t;
  traceName=NULL;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  constructs->append(construct1);
}

SdagConstruct::SdagConstruct(EToken t, SdagConstruct *construct1, SdagConstruct *aList)
{
  con1=0; con2=0; con3=0; con4=0;
  type = t;
  traceName=NULL;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  constructs->append(construct1);
  SdagConstruct *sc;
  for(sc = aList->constructs->begin(); !aList->constructs->end(); sc=aList->constructs->next())
    constructs->append(sc);
}

SdagConstruct::SdagConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
			     SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el)
{
  text = txt;
  type = t;
  traceName=NULL;
  con1 = c1; con2 = c2; con3 = c3; con4 = c4;
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  if (constructAppend != 0) {
    constructs->append(constructAppend);
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
  publishesList = new TList<SdagConstruct*>();
  constructs = new TList<SdagConstruct*>();
  param = pl;

}

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
 
void SdagConstruct::generateConnectEntries(XStr& decls) {
   decls << "  void " <<connectEntry->charstar() << "(";
   ParamList *pl = param;
   XStr msgParams;
   if (pl->isVoid() == 1) {
     decls << "void";
   }
   else if (pl->isMessage() == 1){
     decls << pl->getBaseName() <<" *" <<pl->getGivenName();
   }
   else {
    decls << "CkMarshallMsg *" /*<< connectEntry->charstar()*/ <<"_msg";
   }
   decls << ") {\n";

   if (!pl->isVoid() && !pl->isMessage()) {
    msgParams <<"   char *impl_buf= _msg->msgBuf;\n";
    param->beginUnmarshall(msgParams);
   }

   decls << msgParams.charstar() <<"\n"; 
   decls << "  " <<text->charstar() <<"\n";

   decls << "  }\n";
}

void SdagConstruct::generateConnectEntryList(std::list<SdagConstruct*>& ConnectEList) {
  if (type == SCONNECT)
     ConnectEList.push_back(this);
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
    case SATOMIC:
      generateAtomic(decls, defs, entry);
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
    case SOVERLAP:
      generateOverlap(decls, defs, entry);
      break;
    case SWHEN:
      generateWhen(decls, defs, entry);
      break;
    case SFORWARD:
      generateForward(decls, defs, entry);
      break;
    case SCONNECT:
      generateConnect(decls, defs, entry);
      break;
    default:
      break;
  }
  SdagConstruct *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateCode(decls, defs, entry);
    }
  }
}

void SdagConstruct::generateConnect(XStr& decls, XStr& defs, Entry* entry) {
  generateSignature(decls, defs, entry, false, "void", label, false, NULL);
  defs << "    int index;\n";
  if ((param->isVoid() == 0) && (param->isMessage() == 0)) {
     defs << "    CkMarshallMsg *x;\n";
     defs << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(x);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  else if (param->isVoid() == 1) {
     defs << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(void);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  else {
     defs << "    " << param->getBaseName() <<" *x;\n";  // replace
     defs << "    index = CkIndex_Ar1::" <<connectEntry->charstar() <<"(x);\n";  //replace
     defs << "    CkCallback cb(index, CkArrayIndex1D(thisIndex), a1);\n";  // replace
  }
  defs << "    myPublish->get_" <<connectEntry->charstar() <<"(cb);\n";  //replace - myPublish

  endMethod(defs);
}

void SdagConstruct::generateForward(XStr& decls, XStr& defs, Entry* entry) {
  SdagConstruct *cn;
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  for (cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    defs << "    { ";
    generateCall(defs, *stateVarsChildren, cn->text->charstar());
    defs<<" }\n";
  }
  generateCall(defs, *stateVarsChildren, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}


void SdagConstruct::generateWhen(XStr& decls, XStr& defs, Entry* entry)
{
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateSignature(decls, defs, entry, false, "int", label, false, stateVars);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif

  CStateVar *sv;

  Entry *e;
  EntryList *el;
  el = elist;
  while (el != NULL){
    e = el->entry;
    if (e->param->isVoid() == 1)
        defs << "    CMsgBuffer *"<<e->getEntryName()<<"_buf;\n";
    else if (e->paramIsMarshalled() == 1) {

        defs << "    CMsgBuffer *"<<e->getEntryName()<<"_buf;\n";
        defs << "    CkMarshallMsg *" <<
                        e->getEntryName() << "_msg;\n";
    }
    else {
        for(sv=e->stateVars->begin(); !e->stateVars->end(); e->stateVars->next()) {
          defs << "    CMsgBuffer *"<<sv->name->charstar()<<"_buf;\n";
          defs << "    " << sv->type->charstar() << " " <<
                          sv->name->charstar() << ";\n";
        } 
    }
    el = el->next;
  }

  defs << "\n";
  el = elist;
  while (el != NULL) {
     e = el->entry;

     defs << "    ";
     if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() == 1))
       defs << e->getEntryName();
     else
       defs << sv->name->charstar();
     defs << "_buf = __cDep->getMessage(" << e->entryPtr->entryNum;
     if (e->intExpr)
       defs << ", " << e->intExpr;
     defs << ");\n";

    el = el->next;
  }

  defs << "\n";
  defs << "    if (";
  el = elist;
  while (el != NULL)  {
     e = el->entry;
     if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
        defs << "(" << e->getEntryName() << "_buf != 0)";
     }
     else {
        sv = e->stateVars->begin();
        defs << "(" << sv->name->charstar() << "_buf != 0)";
     }
     el = el->next;
     if (el != NULL)
        defs << "&&";
  }
  defs << ") {\n";

#if CMK_BIGSIM_CHARM
  // for tracing
  //TODO: instead of this, add a length field to EntryList
  int elen=0;
  for(el=elist; el!=NULL; el=elist->next) elen++;
 
  defs << "         void * logs1["<< elen << "]; \n";
  defs << "         void * logs2["<< elen + 1<< "]; \n";
  int localnum = 0;
  for(el=elist; el!=NULL; el=elist->next) {
    e = el->entry;
       if ((e->paramIsMarshalled() == 1) || (e->param->isVoid() ==1)) {
	defs << "       logs1[" << localnum << "] = " << /*el->con4->text->charstar() sv->type->charstar()*/e->getEntryName() << "_buf->bgLog1; \n";
	defs << "       logs2[" << localnum << "] = " << /*el->con4->text->charstar() sv->type->charstar()*/e->getEntryName() << "_buf->bgLog2; \n";
	localnum++;
      }
      else{
	defs << "       logs1[" << localnum << "] = " << /*el->con4->text->charstar()*/ sv->name->charstar()<< "_buf->bgLog1; \n";
	defs << "       logs2[" << localnum << "] = " << /*el->con4->text->charstar()*/ sv->name->charstar() << "_buf->bgLog2; \n";
	localnum++;
      }
  }
      
  defs << "       logs2[" << localnum << "] = " << "_bgParentLog; \n";
  generateEventBracket(defs,SWHEN);
  defs << "       _TRACE_BG_FORWARD_DEPS(logs1,logs2,"<< localnum << ",_bgParentLog);\n";
#endif

  el = elist;
  while (el != NULL) {
     e = el->entry;
     if (e->param->isVoid() == 1) {
        defs <<"       CkFreeSysMsg((void *) "<<e->getEntryName() <<"_buf->msg);\n";
        defs << "       __cDep->removeMessage(" << e->getEntryName() <<
              "_buf);\n";
        defs << "      delete " << e->getEntryName() << "_buf;\n";
     }
     else if (e->paramIsMarshalled() == 1) {
        defs << "       " << e->getEntryName() << "_msg = (CkMarshallMsg *)"
               << e->getEntryName() << "_buf->msg;\n";
        defs << "       char *"<<e->getEntryName() <<"_impl_buf=((CkMarshallMsg *)"
	   <<e->getEntryName() <<"_msg)->msgBuf;\n";
        defs <<"       PUP::fromMem " <<e->getEntryName() <<"_implP("
	   <<e->getEntryName() <<"_impl_buf);\n";

        for(sv=e->stateVars->begin(); !e->stateVars->end(); sv=e->stateVars->next()) {
           if (sv->arrayLength != NULL)
              defs <<"       int impl_off_"<<sv->name->charstar()
	         <<"; "<<e->getEntryName() <<"_implP|impl_off_"
		 <<sv->name->charstar()<<";\n";
           else
               defs <<"       "<<sv->type->charstar()<<" "<<sv->name->charstar()
	       <<"; " <<e->getEntryName() <<"_implP|"
	       <<sv->name->charstar()<<";\n";
	}
        defs << "       " <<e->getEntryName() <<"_impl_buf+=CK_ALIGN("
	   <<e->getEntryName() <<"_implP.size(),16);\n";
        for(sv=e->stateVars->begin(); !e->stateVars->end(); sv=e->stateVars->next()) {
           if (sv->arrayLength != NULL)
              defs << "       "<<sv->type->charstar()<< " *" <<sv->name->charstar() <<"=(" <<sv->type->charstar()
		 <<" *)(" <<e->getEntryName() <<"_impl_buf+" <<"impl_off_"
		 <<sv->name->charstar()<<");\n";
        }
        defs << "       __cDep->removeMessage(" << e->getEntryName() <<
              "_buf);\n";
        defs << "       delete " << e->getEntryName() << "_buf;\n";
     }
     else {  // There was a message as the only parameter
        sv = e->stateVars->begin();
        defs << "       " << sv->name->charstar() << " = (" <<
              sv->type->charstar() << ") " <<
              sv->name->charstar() << "_buf->msg;\n";
        defs << "       __cDep->removeMessage(" << sv->name->charstar() <<
              "_buf);\n";
        defs << "       delete " << sv->name->charstar() << "_buf;\n";
     }
     el = el->next;
  }

  // max(current,merge) --> current, then reset the mergepath
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "       " << label->charstar()  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  defs << "       currentlyExecutingPath = " << label->charstar()  << "_PathMergePoint; /* Critical Path Detection */ \n";
  defs << "       " << label->charstar()  << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  defs << "       ";

  if (constructs && !constructs->empty()) {
    generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  } else {
    generateCall(defs, *stateVarsChildren, label->charstar(), "_end");
  }

  el = elist;
  while (el != NULL){
    e = el->entry;
    if (e->paramIsMarshalled() == 1) {
        defs << "       delete " << e->getEntryName() << "_msg;\n";
    }
    el = el->next;
  }
  defs << "       return 1;\n";
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
  defs << "       CWhenTrigger *tr;\n";
  defs << "       tr = new CWhenTrigger(" << nodeNum << ", " <<
        stateVars->length() << ", " << nRefs << ", " << nAny << ");\n";
  int iArgs=0;
 
//  defs << "       int impl_off=0;\n";
  int hasArray = 0;
  int numParamsNeedingMarshalling = 0;
  int paramIndex =0;
  for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
    if (sv->isVoid == 1) {
       // defs <<"       tr->args[" <<iArgs++ <<"] = (size_t) CkAllocSysMsg();\n";
       defs <<"       tr->args[" <<iArgs++ <<"] = (size_t)0xFF;\n";
    }
    else {
      if (sv->isMsg == 1) {
         defs << "       tr->args["<<iArgs++ <<"] = (size_t) " <<sv->name->charstar()<<";\n";
      }
      else {
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
         defs <<"       int impl_off_"<<sv->name->charstar()<<", impl_cnt_"<<sv->name->charstar()<<";\n";
         defs <<"       impl_off_"<<sv->name->charstar()<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type->charstar()<<"));\n";
         defs <<"       impl_off+=(impl_cnt_"<<sv->name->charstar()<<"=sizeof("<<sv->type->charstar()<<")*("<<sv->arrayLength->charstar()<<"));\n";
      }
    }
  }
  if (numParamsNeedingMarshalling > 0) {
     defs << "       { \n";
     defs << "         PUP::sizer implP;\n";
     for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
       if (sv->arrayLength !=NULL)
         defs << "         implP|impl_off_" <<sv->name->charstar() <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid !=1)) 
         defs << "         implP|" <<sv->name->charstar() <<";\n";
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
     for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
       if (sv->arrayLength !=NULL)
          defs << "         implP|impl_off_" <<sv->name->charstar() <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid != 1))  
          defs << "         implP|" <<sv->name->charstar() <<";\n";
     }
     defs << "       }\n";
     if (hasArray > 0) {
        defs <<"       char *impl_buf=impl_msg->msgBuf+impl_arrstart;\n";
        for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
           if (sv->arrayLength !=NULL)
              defs << "       memcpy(impl_buf+impl_off_"<<sv->name->charstar()<<
	                 ","<<sv->name->charstar()<<",impl_cnt_"<<sv->name->charstar()<<");\n";
        }  
     }
  defs << "       tr->args[" <<paramIndex <<"] = (size_t) impl_msg;\n";
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
  defs << "       " << label->charstar()  << "_PathMergePoint.updateMax(currentlyExecutingPath); /* Critical Path Detection */ \n";
  defs << "       currentlyExecutingPath = " << label->charstar()  << "_PathMergePoint; /* Critical Path Detection */ \n";
#endif

  defs << "       __cDep->Register(tr);\n";
  defs << "       return 0;\n";
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
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  
  el = elist;
  while (el) {
    e = el->entry;
    if (e->param->isMessage() == 1) {
      sv = e->stateVars->begin();
      defs << "    CmiFree(UsrToEnv(" << sv->name->charstar() << "));\n";
    }

    el = el->next;
  }

  endMethod(defs);
}

void SdagConstruct::generateWhile(XStr& decls, XStr& defs, Entry* entry)
{
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    if (" << con1->text->charstar() << ") {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    } else {\n";
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
  defs << "    if (" << con1->text->charstar() << ") {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    } else {\n";
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);
}

void SdagConstruct::generateFor(XStr& decls, XStr& defs, Entry* entry)
{
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());

  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif
  defs << "    " << con1->text->charstar() << ";\n";
  //Record only the beginning for FOR
#if CMK_BIGSIM_CHARM
  generateEventBracket(defs, SFOR);
#endif
  defs << "    if (" << con2->text->charstar() << ") {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    } else {\n";
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
#endif
  defs << "   " << con3->text->charstar() << ";\n";
  defs << "    if (" << con2->text->charstar() << ") {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    } else {\n";
#if CMK_BIGSIM_CHARM
  generateEventBracket(defs, SFOR_END);
#endif
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);
}

void SdagConstruct::generateIf(XStr& decls, XStr& defs, Entry* entry)
{
  strcpy(nameStr,label->charstar());
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs, SIF);
#endif
  defs << "    if (" << con1->text->charstar() << ") {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    } else {\n";
  defs << "      ";
  if (con2 != 0) {
    generateCall(defs, *stateVarsChildren, con2->label->charstar());
  } else {
    generateCall(defs, *stateVarsChildren, label->charstar(), "_end");
  }
  defs << "    }\n";
  endMethod(defs);

  strcpy(nameStr,label->charstar());
  strcat(nameStr,"_end");
  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs,SIF_END);
#endif
  defs << "    ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void SdagConstruct::generateElse(XStr& decls, XStr& defs, Entry* entry)
{
  strcpy(nameStr,label->charstar());
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  // trace
  generateBeginTime(defs);
  generateEventBracket(defs, SELSE);
  defs << "    ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  endMethod(defs);

  // trace
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  generateEventBracket(defs,SELSE_END);
#endif
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void SdagConstruct::generateForall(XStr& decls, XStr& defs, Entry* entry)
{
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    int __first = (" << con2->text->charstar() <<
        "), __last = (" << con3->text->charstar() << 
        "), __stride = (" << con4->text->charstar() << ");\n";
  defs << "    if (__first > __last) {\n";
  defs << "      int __tmp=__first; __first=__last; __last=__tmp;\n";
  defs << "      __stride = -__stride;\n";
  defs << "    }\n";
  defs << "    CCounter *" << counter->charstar() <<
        " = new CCounter(__first,__last,__stride);\n"; 
  defs << "    for(int " << con1->text->charstar() <<
        "=__first;" << con1->text->charstar() <<
        "<=__last;" << con1->text->charstar() << "+=__stride) {\n";
  defs << "      ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  defs << "    }\n";
  endMethod(defs);

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
  defs << "    " << counter->charstar() << "->decrement(); /* DECREMENT 1 */ \n";
  defs << "    if (" << counter->charstar() << "->isDone()) {\n";
  defs << "      delete " << counter->charstar() << ";\n";
  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  defs << "    }\n";
  endMethod(defs);
}

void SdagConstruct::generateOlist(XStr& decls, XStr& defs, Entry* entry)
{
  SdagConstruct *cn;
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    CCounter *" << counter->charstar() << "= new CCounter(" <<
        constructs->length() << ");\n";
  for(cn=constructs->begin(); 
                     !constructs->end(); cn=constructs->next()) {
    defs << "    ";
    generateCall(defs, *stateVarsChildren, cn->label->charstar());
  }
  endMethod(defs);

  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  strcat(nameStr,"_end");
#if CMK_BIGSIM_CHARM
  defs << "  CkVec<void*> " <<label->charstar() << "_bgLogList;\n";
#endif

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
#if CMK_BIGSIM_CHARM
  generateBeginTime(defs);
  defs << "    " <<label->charstar() << "_bgLogList.insertAtEnd(_bgParentLog);\n";
#endif
  //Accumulate all the bgParent pointers that the calling when_end functions give
  defs << "    " << counter->charstar() << "->decrement();\n";
 
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
 defs << "    olist_" << counter->charstar() << "_PathMergePoint.updateMax(currentlyExecutingPath);  /* Critical Path Detection FIXME: is the currently executing path the right thing for this? The duration ought to have been added somewhere. */ \n";
#endif

  defs << "    if (" << counter->charstar() << "->isDone()) {\n";

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "      currentlyExecutingPath = olist_" << counter->charstar() << "_PathMergePoint; /* Critical Path Detection */ \n";
  defs << "      olist_" << counter->charstar() << "_PathMergePoint.reset(); /* Critical Path Detection */ \n";
#endif

  defs << "      delete " << counter->charstar() << ";\n";

#if CMK_BIGSIM_CHARM
  generateListEventBracket(defs, SOLIST_END);
  defs << "       "<< label->charstar() <<"_bgLogList.length()=0;\n";
#endif

  defs << "      ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
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
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
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
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void SdagConstruct::generateSlist(XStr& decls, XStr& defs, Entry* entry)
{
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);
  defs << "    ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());
  endMethod(defs);

  generateSignature(decls, defs, entry, false, "void", label, true, stateVarsChildren);
  defs << "    ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void SdagConstruct::generateSdagEntry(XStr& decls, XStr& defs, Entry *entry)
{
  if (entry->isConstructor()) {
    std::cerr << cur_file << ":" << entry->getLine()
              << ": Chare constructor cannot be defined with SDAG code" << std::endl;
    exit(1);
  }

  decls << "public:\n";
  generateSignature(decls, defs, entry, false, "void", con1->text, false, stateVars);
  SdagConstruct *sc;
  SdagConstruct *sc1;
  for(sc =publishesList->begin(); !publishesList->end(); sc=publishesList->next()) {
     for(sc1=sc->constructs->begin(); !sc->constructs->end(); sc1 = sc->constructs->next())
        defs << "    _connect_" << sc1->text->charstar() <<"();\n";
  }

#if CMK_BIGSIM_CHARM
  generateEndSeq(defs);
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor())
    generateTraceEndCall(defs);

  defs << "    ";
  generateCall(defs, *stateVarsChildren, constructs->front()->label->charstar());

#if CMK_BIGSIM_CHARM
  generateTlineEndCall(defs);
  generateBeginExec(defs, "spaceholder");
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor())
    generateDummyBeginExecute(defs);

  endMethod(defs);

  decls << "private:\n";
  generateSignature(decls, defs, entry, false, "void", con1->text, true,
#if CMK_BIGSIM_CHARM
  stateVarsChildren
#else
  stateVars
#endif
);
  endMethod(defs);
}

void SdagConstruct::generateAtomic(XStr& decls, XStr& defs, Entry* entry)
{
  generateSignature(decls, defs, entry, false, "void", label, false, stateVars);

#if CMK_BIGSIM_CHARM
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateBeginExec(defs, nameStr);
#endif
  generateTraceBeginCall(defs);

  defs << "    " << text->charstar() << "\n";

  generateTraceEndCall(defs);
#if CMK_BIGSIM_CHARM
  generateEndExec(defs);
#endif

  defs << "    ";
  generateCall(defs, *stateVars, next->label->charstar(), nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void generateSignature(XStr& str,
                       const XStr* name, const char* suffix,
                       TList<CStateVar*>* params)
{

}
void generateSignature(XStr& decls, XStr& defs,
                       const Entry* entry, bool declareStatic, const char* returnType,
                       const XStr* name, bool isEnd,
                       TList<CStateVar*>* params)
{
  generateSignature(decls, defs, entry->getContainer(), declareStatic, returnType,
                    name, isEnd, params);
}
void generateSignature(XStr& decls, XStr& defs,
                       const Chare* chare, bool declareStatic, const char* returnType,
                       const XStr* name, bool isEnd,
                       TList<CStateVar*>* params)
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
    for (sv = params->begin(); !params->end(); ) {
      if (sv->isVoid != 1) {
        if (count != 0)
          op << ", ";

        if (sv->type != 0) 
          op <<sv->type->charstar() <<" ";
        if (sv->byRef != 0)
          op <<" &";
        if (sv->arrayLength != NULL) 
          op <<"* ";
        if (sv->name != 0)
          op <<sv->name->charstar();

        count++;
      }

      sv = params->next();
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
    op << "    " << "_TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (" << "_sdag_idx_" << traceName->charstar() << "()), CkMyPe(), 0, NULL); \n";
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

void SdagConstruct::generateRegisterEp(XStr& defs)
{
  if (traceName) {
    defs << "    (void)_sdag_idx_" << traceName << "();\n";
  }

  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateRegisterEp(defs);
  }
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

  SdagConstruct *cn;
  for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    cn->generateTraceEp(decls, defs, chare);
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
