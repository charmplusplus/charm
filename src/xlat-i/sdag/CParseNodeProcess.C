/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/
#include <string.h>
#include <stdlib.h>
#include "sdag-globals.h"
#include "CParseNode.h"
void CParseNode::numberNodes(void)
{
  switch(type) {
    case SDAGENTRY: nodeNum = numSdagEntries++; break;
    case OVERLAP: nodeNum = numOverlaps++; break;
    case WHEN: nodeNum = numWhens++; break;
    case FOR: nodeNum = numFors++; break;
    case WHILE: nodeNum = numWhiles++; break;
    case IF: nodeNum = numIfs++; if(con2!=0) con2->numberNodes(); break;
    case ELSE: nodeNum = numElses++; break;
    case FORALL: nodeNum = numForalls++; break;
    case SLIST: nodeNum = numSlists++; break;
    case OLIST: nodeNum = numOlists++; break;
    case ATOMIC: nodeNum = numAtomics++; break;
    case FORWARD: nodeNum = numForwards++; break;
    case ELIST:
    case INT_EXPR:
    case IDENT: 
    case ENTRY:
    default:
      break;
  }
  CParseNode *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->numberNodes();
    }
  }
}

void CParseNode::labelNodes(void)
{
  char text[128];
  switch(type) {
    case ENTRY:
    case SDAGENTRY:
      sprintf(text, "%s", con1->text->charstar());
      label = new XStr(text);
      break;
    case OVERLAP: 
      sprintf(text, "_overlap_%d", nodeNum); 
      label = new XStr(text);
      break;
    case WHEN: 
      sprintf(text, "_when_%d", nodeNum); 
      label = new XStr(text);
      break;
    case FOR: 
      sprintf(text, "_for_%d", nodeNum); 
      label = new XStr(text);
      break;
    case WHILE: 
      sprintf(text, "_while_%d", nodeNum); 
      label = new XStr(text);
      break;
    case IF: 
      sprintf(text, "_if_%d", nodeNum); 
      label = new XStr(text);
      if(con2!=0) con2->labelNodes();
      break;
    case ELSE: 
      sprintf(text, "_else_%d", nodeNum); 
      label = new XStr(text);
      break;
    case FORALL: 
      sprintf(text, "_forall_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SLIST: 
      sprintf(text, "_slist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case OLIST: 
      sprintf(text, "_olist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case ATOMIC: 
      sprintf(text, "_atomic_%d", nodeNum); 
      label = new XStr(text);
      break;
    case FORWARD: 
      sprintf(text, "_forward_%d", nodeNum); 
      label = new XStr(text);
      break;
    case ELIST:
    case INT_EXPR:
    case IDENT:
    default:
      break;
  }
  CParseNode *cn;
  if (constructs != 0) {
    for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
      cn->labelNodes();
    }
  }
}

void CParseNode::generateEntryList(TList<CEntry*>& elist, TList<COverlap*>& olist, CParseNode *thisWhen)
{
  CParseNode *cn;
  int hasOverlapList;
  switch(type) {
    case OLIST:
      hasOverlapList = 0;
      COverlap *overlap;
       overlap = new COverlap(nodeNum);
      for (cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
         if(cn->type == WHEN) {
	    hasOverlapList = 1;
            overlap->whenList.append(cn);
	 }
      }
      if (hasOverlapList == 1)
         olist.append(overlap);
      break;
    case WHEN:
      con1->generateEntryList(elist, olist, this);  /* con1 is the WHEN's ELIST */
      break;
    case IF:
	/* con2 is the ELSE corresponding to this IF */
      if(con2!=0) con2->generateEntryList(elist,olist, thisWhen); 
      break;
    case ENTRY:
      CEntry *entry;
      int notfound=1;
      for(entry=elist.begin(); !elist.end(); entry=elist.next()) {
     	if(*(entry->entry) == *(con1->text)) 
        {
           CParseNode *param1;
           CParseNode *param2;
	   param1 = entry->paramlist->constructs->begin();
           param2 = con3->constructs->begin();
           notfound = 1;
	   if ((entry->paramlist->isVoid == 1) && (con3->isVoid == 1)) {
	      notfound = 0;
	   }
	   while ((notfound == 1) && (!entry->paramlist->constructs->end()) && (!con3->constructs->end()))
	   {
              if(*(param1->vartype) == *(param2->vartype)) {
	        if ((param1->con3 != 0) && (param2->con3 != 0)) {
		   if (*(param1->con3->text) == *(param2->con3->text)) {
		      notfound = 0;
		   }
		}
		else {
                  notfound = 0;
                }
              }
 	      param1 = entry->paramlist->constructs->next(); 
              param2 = con3->constructs->next();
           }
           if (((!con3->constructs->end()) && (entry->paramlist->constructs->end())) ||
              ((con3->constructs->end()) && (!entry->paramlist->constructs->end())))
           {
		notfound = 1;
	   }
	   if (notfound == 0) {
             // check to see if thisWhen is already in entry's whenList
             int whenFound = 0;
             TList<CParseNode*> *tmpList = &(entry->whenList);
             CParseNode *tmpNode;
             for(tmpNode = tmpList->begin(); !tmpList->end(); tmpNode = tmpList->next()) {
               if(tmpNode->nodeNum == thisWhen->nodeNum)
                  whenFound = 1;
             }
             if(!whenFound)
               entry->whenList.append(thisWhen);
             entryPtr = entry;
             if(con2)
               entry->refNumNeeded = 1;
             break;
	   } 
        }
      }
      if(notfound == 1) {
        CEntry *newEntry;
        newEntry = new CEntry(new XStr(*(con1->text)), con3, estateVars, needsParamMarshalling );
        elist.append(newEntry);
        entryPtr = newEntry;
        newEntry->whenList.append(thisWhen);
        if(con2)
          newEntry->refNumNeeded = 1;
      }
      break;
  }
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateEntryList(elist,olist,thisWhen);
    }
  }
}


void CParseNode::propagateState(int uniqueVarNum)
{ 
  int count = 0;
  int isMsg = 0;
  int isVoidParameter = 0;
  int numParameters = 0;
  needsParamMarshalling = 0;
  CStateVar *sv; 
  if(type != SDAGENTRY) {
    fprintf(stderr, "use of non-entry as the outermost construct..\n");
    exit(1);
  }
  stateVars = new TList<CStateVar*>();

  CParseNode *parameter1;
  CParseNode *vartype1;

  if (con2->isVoid == 1) {
     sv = new CStateVar(0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
     stateVars->append(sv);
  }
  else { 
    // Traverses through the SDAGENTRY's parameter list 
    for (parameter1=con2->constructs->begin(); !con2->constructs->end(); parameter1=con2->constructs->next()) { 
        if ((numParameters == 1) && (isMsg == 1))
           printf("ERROR: When you have a message as a parameter you can not have any other parameters\n");
   	   
        numParameters++;
        XStr *vartype = new XStr("");
        vartype1 = parameter1->con1;
        XStr *constStr;
        count = 0;
        while (vartype1->con1 !=0) {
           if (count == 0)
	      constStr = new XStr("const ");
	   else
              constStr->append("const ");
	   vartype1 = vartype1->con2;
	   vartype->append("const ");
	   count++;
        }
        XStr *byRef = 0;
        if (parameter1->con1->con4 != 0) {
          if (parameter1->con1->con4->type == AMPERESIGN) {
	    byRef = new XStr("&");
	  }
          else
             byRef = 0;
        } 
        if (count == 0)
           constStr = 0;
        if (vartype1->con3->type == SIMPLETYPE) {
	   isMsg = 0;
           needsParamMarshalling =1;
	   XStr *vType1 = new XStr(*(vartype1->con3->con1->con1->text));
	   vartype->append(*(vartype1->con3->con1->con1->text));
           XStr *vType2;
           if (vartype1->con3->con1->con2 != 0) {
	      vType2 = new XStr(*(vartype1->con3->con1->con2->text));	
	      vartype->append(" ");
              vartype ->append(*(vartype1->con3->con1->con2->text)); 
           }
	   else 
             vType2 = 0;
	   XStr *name;
           if ((parameter1->con2 == 0) && (parameter1->con3 == 0)) {  // Type
	       name = new XStr("noname_"); *name<<uniqueVarNum;
	       uniqueVarNum++;
               sv = new CStateVar(constStr, 0, vType1, vType2, 0, 0, name, byRef, 0, isMsg);
	   }
	   else if ((parameter1->con2 != 0) && (parameter1->con3 == 0))    // Type Name
               sv = new CStateVar(constStr, 0,vType1, vType2, 0, 0, new XStr(*(parameter1->con2->text)),byRef,0, isMsg);
	   else if ((parameter1->con2 != 0) && (parameter1->con3 != 0))   // Type Name [ArrayLength]
	       sv = new CStateVar(constStr, 0,vType1, vType2, 0, 0, new XStr(*(parameter1->con2->text)), byRef,new XStr(*(parameter1->con3->text)), isMsg);
	   stateVars->append(sv);
        }
        else if (vartype1->con3->type == PTRTYPE) {
           isMsg = 0;
           XStr *vType1 = new XStr(*(vartype1->con3->con1->con1->con1->text));
           vartype->append(*(vartype1->con3->con1->con1->con1->text));
           XStr *vType2;
           if (vartype1->con3->con1->con1->con2 != 0) {
              vType2 = new XStr(*(vartype1->con3->con1->con1->con2->text));
              vartype->append(" ");
              vartype->append(*(vartype1->con3->con1->con1->con2->text));
           }
           else
              vType2 = 0;
           XStr *pt = new XStr("*");
           for(count=1; count<vartype1->con3->numPtrs; count++)
              pt->append("*");
           vartype->append(*(pt));
	   if ((numParameters == 1) && (vartype1->con3->numPtrs == 1))
	      isMsg = 1;
	       
	      XStr *name;
	   if ((parameter1->con2 == 0) && (parameter1->con3 == 0)) {
	      name = new XStr("noname_"); *name<<uniqueVarNum;
	      uniqueVarNum++;
              sv = new CStateVar(constStr, 0, vType1, vType2, pt, vartype1->con3->numPtrs, name,byRef, 0, isMsg);
	   }
	   else if ((parameter1->con2 != 0) && (parameter1->con3 == 0)) 
              sv = new CStateVar(constStr,0, vType1, vType2, pt, vartype1->con3->numPtrs, new XStr(*(parameter1->con2->text)), byRef, 0, isMsg);
	   else if ((parameter1->con2 != 0) && (parameter1->con3 != 0)) 
              sv = new CStateVar(constStr, 0, vType1, vType2, pt, vartype1->con3->numPtrs, new XStr(*(parameter1->con2->text)), byRef, new XStr(*(parameter1->con3->text)), isMsg);
           stateVars->append(sv);
        }
        if (byRef != 0) {
           vartype->append(" &"); 
	}
	parameter1->vartype = vartype;
    }

  } 
  stateVarsChildren = stateVars; 
  CParseNode *cn;
  TList<CStateVar*> *whensEntryMethodStateVars; 
  whensEntryMethodStateVars = new TList<CStateVar*>();
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->propagateState(*stateVarsChildren, *whensEntryMethodStateVars , uniqueVarNum);
    }
  }

}

void CParseNode::propagateState(TList<CStateVar*>& list, TList<CStateVar*>& wlist, int uniqueVarNum)
{
  CStateVar *sv;
  int i;
  TList <CStateVar*> *olistTempStateVars;
  TList<CStateVar*> *whensEntryMethodStateVars; 
  olistTempStateVars = new TList<CStateVar*>();
  stateVars = new TList<CStateVar*>();
  switch(type) {
    case FORALL:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      sv = new CStateVar(0,0, new XStr("int"), 0, 0, 0, new XStr(*(con1->text)), 0,0, 0);
      stateVarsChildren->append(sv);
      {
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new XStr(txt);
        XStr *ptrs = new XStr("*");
        sv = new CStateVar(0, 0, new XStr("CCounter" ), 0, ptrs, 1, counter,0,0, 1);
        stateVarsChildren->append(sv);
      }
      break;
    case WHEN:
      whensEntryMethodStateVars = new TList<CStateVar*>();
      stateVarsChildren = new TList<CStateVar*>();
      int numParameters; 
      int count;
      int isMsg;
      int stateVarsHasVoid;
      int stateVarsChildrenHasVoid; 
      numParameters=0; count=0; isMsg=0; 
      stateVarsHasVoid = 0;
      stateVarsChildrenHasVoid = 0;
      for(sv = stateVars->begin(); ((!stateVars->end()) && (stateVarsHasVoid != 1)); sv=stateVars->next()) {
         if (sv->isVoid == 1)
	     stateVarsHasVoid == 1;
      }
      for(sv=list.begin(); !list.end(); sv=list.next()) {
         if ((sv->isVoid == 1) && (stateVarsHasVoid != 1)) {
	    stateVars->append(sv);
	    stateVarsHasVoid == 1;
	 }
	 else if (sv->isVoid != 1)
	    stateVars->append(sv);
	 if ((sv->isVoid == 1) && (stateVarsChildrenHasVoid != 1)) {
	    stateVarsChildren->append(sv);
	    stateVarsChildrenHasVoid == 1;
	 }
	 else if (sv->isVoid != 1) {
	    stateVarsChildren->append(sv);
	 }
      }

     
      {  
        TList<CParseNode*> *elist = con1->constructs;
        CParseNode *entry1;
        CParseNode *parameter1;
        CParseNode *type1;
        CParseNode *type2;

// Traverses the entry list 
        for(entry1=elist->begin(); !elist->end(); entry1=elist->next()) { 
 
// Traverses the parameters for each entry method 
            entry1->stateVars = new TList<CStateVar*>();
            numParameters = 0; entry1->needsParamMarshalling = 0;

            if (entry1->con3->isVoid == 1) { 
	       entry1->isVoid = 1;
               sv = new CStateVar(0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
	       if (stateVarsChildrenHasVoid != 1) {
                  stateVarsChildren->append(sv);
	       }
 	       entry1->stateVars->append(sv);
               whensEntryMethodStateVars->append(sv); 
       	       entry1->estateVars.append(sv);
            }
            else { 
  	      for (parameter1=entry1->con3->constructs->begin(); !entry1->con3->constructs->end(); 
	  	  parameter1=entry1->con3->constructs->next()) { 
		  if ((numParameters == 1) && (isMsg == 1))
                     printf("ERROR: When you have a message as a parameter you can not have any other parameters in an entry function\n");
		 
                  numParameters++;
                  XStr *vartype = new XStr("");
		  XStr *byRef = 0;
 	          type1 = parameter1->con1;
		  XStr *constStr= 0;
                  count = 0;
		  while (type1->con1 != 0) {
		     if (count == 0)
		        constStr = new XStr("const ");
		     else
		        constStr->append("const ");
	             vartype->append("const ");
		     count++;
		     type1 = type1->con2;
		  }
                  if (parameter1->con1->con4 != 0) {
                    if (parameter1->con1->con4->type == AMPERESIGN) {
		      byRef = new XStr("&");
		    }
                    else
                       byRef = 0; 
                  }
		  if (type1->con3->type == SIMPLETYPE) {
		     isMsg = 0;
                     entry1->needsParamMarshalling =1;
	             XStr *vType1 = new XStr(*(type1->con3->con1->con1->text)); 
	             vartype->append(*(type1->con3->con1->con1->text));
     		     XStr *vType2;
                     if (type1->con3->con1->con2 != 0) {
		        vType2 = new XStr(*(type1->con3->con1->con2->text));	
	                vartype->append(" ");
                        vartype->append(*(type1->con3->con1->con2->text));
		     }	
		     else 
		        vType2 = 0;
	             XStr *name;
	             if ((parameter1->con2 == 0) && (parameter1->con3 == 0)) {
	                name = new XStr("noname_"); *name<<uniqueVarNum;
	                uniqueVarNum++;
                        sv = new CStateVar(constStr, 0, vType1, vType2, 0, 0, name, byRef, 0, isMsg);
		     }
	             else if ((parameter1->con2 != 0) && (parameter1->con3 == 0)) 
                        sv = new CStateVar(constStr, 0, vType1, vType2, 0, 0, new XStr(*(parameter1->con2->text)),byRef,0, isMsg);
		     else if ((parameter1->con2 != 0) && (parameter1->con3 != 0)) 
                        sv = new CStateVar(constStr, 0, vType1, vType2, 0,0 , new XStr(*(parameter1->con2->text)), byRef,new XStr(*(parameter1->con3->text)), isMsg);
                     stateVarsChildren->append(sv);
                     whensEntryMethodStateVars->append(sv); 
 		     entry1->estateVars.append(sv);
 		     entry1->stateVars->append(sv);
		  }
		  else if (type1->con3->type == PTRTYPE) { 
		     isMsg = 0;
                     XStr *vType1 = new XStr(*(type1->con3->con1->con1->con1->text));
                     vartype->append(*(type1->con3->con1->con1->con1->text));
                     XStr *vType2;
                     if (type1->con3->con1->con1->con2 != 0) {
                        vType2 = new XStr(*(type1->con3->con1->con1->con2->text));
                        vartype->append(" ");
                        vartype->append(*(type1->con3->con1->con1->con2->text));
		     }
                     else
                        vType2 = 0;
                     XStr *pt = new XStr("*");
                     for(i=1; i<type1->con3->numPtrs; i++)
                        pt->append("*");
                     vartype->append(*(pt));
	             if ((numParameters == 1) && (type1->con3->numPtrs == 1))
	                isMsg = 1;
	                XStr *name;
	             if ((parameter1->con2 == 0) && (parameter1->con3 == 0)) {
	                name = new XStr("noname_"); *name<<uniqueVarNum;
	                uniqueVarNum++;
                        sv = new CStateVar(constStr, 0,  vType1, vType2, pt, type1->con3->numPtrs, name,byRef, 0, isMsg);
		     }
	             else if ((parameter1->con2 != 0) && (parameter1->con3 == 0)) 
                        sv = new CStateVar(constStr, 0, vType1, vType2, pt, type1->con3->numPtrs, 
                                           new XStr(*(parameter1->con2->text)),byRef,0, isMsg);
		     else if ((parameter1->con2 != 0) && (parameter1->con3 != 0))  
                        sv = new CStateVar(constStr, 0, vType1, vType2, pt, type1->con3->numPtrs, 
                                           new XStr(*(parameter1->con2->text)), byRef, 
                                           new XStr(*(parameter1->con3->text)), isMsg);
                     stateVarsChildren->append(sv);
 	             entry1->stateVars->append(sv);
                     whensEntryMethodStateVars->append(sv); 
       	             entry1->estateVars.append(sv);
		  }
                  if (byRef != 0) {
                    vartype->append(" &"); 
		  }  
                  parameter1->vartype = vartype;
	      }
	    } 

          if ((entry1->needsParamMarshalling == 0) && (numParameters >1))
	     printf("Error: There is a problem with the parameters of the entry list\n");  
        } 
      }
      break;
    case IF:
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      if(con2 != 0) con2->propagateState(list, wlist, uniqueVarNum);
      break;
    case OLIST:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      {
        char txt[128];
        sprintf(txt, "_co%d", nodeNum);
        counter = new XStr(txt);
	XStr *ptrs = new XStr("*");
        sv = new CStateVar(0,0, new XStr("CCounter"), 0, ptrs, 1, counter,0, 0, 1);
        stateVarsChildren->append(sv);
      }
      break;
    case FOR:
    case WHILE:
    case ELSE:
    case SLIST:
    case OVERLAP:
    case ATOMIC:
      for(sv=list.begin(); !list.end(); sv=list.next()) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      break;
    case FORWARD:
      stateVarsChildren = new TList<CStateVar*>();
      for(sv=list.begin(); !list.end(); sv=list.next()) { 
        stateVars->append(sv);
      }
      for(sv=wlist.begin(); !wlist.end(); sv=wlist.next()) { 
        stateVarsChildren->append(sv);
      }

      break;
    case INT_EXPR:
    case IDENT:
    case ENTRY:
    case ELIST:
      break;
    default:
      fprintf(stderr, "internal error in sdag translator..\n");
      exit(1);
      break;
  }
  CParseNode *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      if (type == WHEN)
         cn->propagateState(*stateVarsChildren, *whensEntryMethodStateVars,  uniqueVarNum);
      else
         cn->propagateState(*stateVarsChildren, wlist,  uniqueVarNum);
    }
 } 
}

void CParseNode::generateCode(XStr& op)
{
  switch(type) {
    case SDAGENTRY:
      generateSdagEntry(op);
      break;
    case SLIST:
      generateSlist(op);
      break;
    case OLIST:
      generateOlist(op);
      break;
    case FORALL:
      generateForall(op);
      break;
    case ATOMIC:
      generateAtomic(op);
      break;
    case IF:
      generateIf(op);
      if(con2 != 0)
        con2->generateCode(op);
      break;
    case ELSE:
      generateElse(op);
      break;
    case WHILE:
      generateWhile(op);
      break;
    case FOR:
      generateFor(op);
      break;
    case OVERLAP:
      generateOverlap(op);
      break;
    case WHEN:
      generateWhen(op);
      break;
    case FORWARD:
      generateForward(op);
      break;
    default:
      break;
  }
  CParseNode *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->generateCode(op);
    }
  }
}

void CParseNode::generateForward(XStr& op) {
  CParseNode *cn;
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  for (cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
    op << "    { "<<cn->text->charstar()<<"(";
    generateCall(op, *stateVarsChildren);
    op<<"); }\n";
  }
  if(nextBeginOrEnd == 1)
    op << "    " << next->label->charstar() << "(";
  else
    op << "    " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  op << "  }\n\n";
}
void CParseNode::generateWhen(XStr& op)
{
  op << "  int " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  TList<CParseNode*> *elist = con1->constructs;
  CParseNode *el;

  CStateVar *sv;
  for(el=elist->begin(); !elist->end(); el=elist->next()) {
     if (el->isVoid == 1) {
         op << "    CMsgBuffer *"<<el->con1->text->charstar()<<"_buf;\n";
     }
     else if (el->needsParamMarshalling == 1) {
         op << "    CMsgBuffer *"<<el->con1->text->charstar()<<"_buf;\n";
         op << "    CkMarshallMsg *" <<
                         el->con1->text->charstar() << "_msg;\n";
     }
     else {
       for(sv=el->stateVars->begin(); !el->stateVars->end(); el->stateVars->next()) {
         op << "    CMsgBuffer *"<<sv->name->charstar()<<"_buf;\n";
         op << "    " << sv->type1->charstar() << " *" <<
                         sv->name->charstar() << ";\n";
       } 
     }
  }


  op << "\n";
  for(el=elist->begin(); !elist->end(); el=elist->next()) {
     if ((el->needsParamMarshalling == 1) || (el->isVoid == 1)) {
        if((el->con2 == 0) || (el->isVoid == 1)) {
           op << "    " << el->con1->text->charstar(); 
           op << "_buf = __cDep->getMessage(" << el->entryPtr->entryNum << ");\n";
        }	    
        else {
           op << "    " << el->con1->text->charstar() << 
                 "_buf = __cDep->getMessage(" << el->entryPtr->entryNum <<
                 ", " << el->con2->text->charstar() << ");\n";
        }
     }
     else { // The parameter is a message
        sv = el->stateVars->begin();
        if(el->con2 == 0) {
           op << "    " << sv->name->charstar(); 
           op << "_buf = __cDep->getMessage(" << el->entryPtr->entryNum << ");\n";
        }	    
        else {
           op << "    " << sv->name->charstar() << 
                 "_buf = __cDep->getMessage(" << el->entryPtr->entryNum <<
                 ", " << el->con2->text->charstar() << ");\n";
        }
     }
  }
  op << "\n";
  op << "    if (";
  for(el=elist->begin(); !elist->end();) {
     if ((el->needsParamMarshalling == 1) || (el->isVoid ==1)) {
        op << "(" << el->con1->text->charstar() << "_buf != 0)";
     }
     else {
        sv = el->stateVars->begin();
        op << "(" << sv->name->charstar() << "_buf != 0)";
     }
     el = elist->next();
     if(el != 0)
        op << "&&";
  }
  op << ") {\n";
  for(el=elist->begin(); !elist->end(); el=elist->next()) {
     if (el->isVoid == 1) {
        op <<"       CkFreeSysMsg((void *) "<<el->con1->text->charstar() <<"_buf->msg);\n";
        op << "      delete " << el->con1->text->charstar() << "_buf;\n";
     }
     else if (el->needsParamMarshalling == 1) {
        op << "       " << el->con1->text->charstar() << "_msg = (CkMarshallMsg *)"  
               << el->con1->text->charstar() << "_buf->msg;\n";
        op << "       char *"<<el->con1->text->charstar() <<"_impl_buf=((CkMarshallMsg *)"
	   <<el->con1->text->charstar() <<"_msg)->msgBuf;\n";
        op <<"       PUP::fromMem " <<el->con1->text->charstar() <<"_implP(" 
	   <<el->con1->text->charstar() <<"_impl_buf);\n";

        for(sv=el->stateVars->begin(); !el->stateVars->end(); sv=el->stateVars->next()) {
           if (sv->arrayLength != 0)
              op <<"      int impl_off_"<<sv->name->charstar()
	         <<"; "<<el->con1->text->charstar() <<"_implP|impl_off_"
		 <<sv->name->charstar()<<";\n";
           else
               op <<"       "<<sv->type1->charstar()<<" "<<sv->name->charstar()
	       <<"; " <<el->con1->text->charstar() <<"_implP|"
	       <<sv->name->charstar()<<";\n";
	}
	
        op << "       " <<el->con1->text->charstar() <<"_impl_buf+=CK_ALIGN("
	   <<el->con1->text->charstar() <<"_implP.size(),16);\n";
        for(sv=el->stateVars->begin(); !el->stateVars->end(); sv=el->stateVars->next()) {
           if (sv->arrayLength != 0)
              op << "    "<<sv->type1->charstar()<< " *" <<sv->name->charstar() <<"=(" <<sv->type1->charstar()
		 <<" *)(" <<el->con1->text->charstar() <<"_impl_buf+" <<"impl_off_"
		 <<sv->name->charstar()<<");\n";
        }
        op << "       __cDep->removeMessage(" << el->con1->text->charstar() <<
              "_buf);\n";
        op << "       delete " << el->con1->text->charstar() << "_buf;\n";

     }
     else {  // There was a message as the only parameter
        sv = el->stateVars->begin();
        op << "       " << sv->name->charstar() << " = (" << 
              sv->type1->charstar() << " *) " <<
              sv->name->charstar() << "_buf->msg;\n";
        op << "       __cDep->removeMessage(" << sv->name->charstar() <<
              "_buf);\n";
        op << "       delete " << sv->name->charstar() << "_buf;\n";
     }
  }
  if (!constructs->empty() ) {
     op << "       " << constructs->front()->label->charstar() << "(";
     generateCall(op, *stateVarsChildren);
     op << ");\n";
  }
  else {
     op << "       " << label->charstar() << "_end(";
     generateCall(op, *stateVarsChildren);
     op << ");\n";
  }
  op << "       return 1;\n";

  op << "    } else {\n";

  int nRefs=0, nAny=0;
  for(el=elist->begin(); !elist->end(); el=elist->next()) {
    if(el->con2 == 0)
      nAny++;
    else
      nRefs++;
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
  int isVoid = 0;
  int numParamsNeedingMarshalling = 0;
  int paramIndex =0;
  for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
    if (sv->isVoid == 1) {
        isVoid = 1;
       op <<"       tr->args[" <<iArgs++ <<"] = (size_t) CkAllocSysMsg();\n";
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
      if (sv->arrayLength !=0) {
         hasArray++;
         if (hasArray == 1)
      	   op<< "       int impl_arrstart=0;\n";
         op <<"       int impl_off_"<<sv->name->charstar()<<", impl_cnt_"<<sv->name->charstar()<<";\n";
         op <<"       impl_off_"<<sv->name->charstar()<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type1->charstar()<<"));\n";
         op <<"       impl_off+=(impl_cnt_"<<sv->name->charstar()<<"=sizeof("<<sv->type1->charstar()<<")*("<<sv->arrayLength->charstar()<<"));\n";
      }
    }
  }
  if (numParamsNeedingMarshalling > 0) {
     op << "       { \n";
     op << "         PUP::sizer implP;\n";
     for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
       if (sv->arrayLength !=0)
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
       if (sv->arrayLength !=0)
          op << "         implP|impl_off_" <<sv->name->charstar() <<";\n";
       else if ((sv->isMsg != 1) && (sv->isVoid != 1))  
          op << "         implP|" <<sv->name->charstar() <<";\n";
     }
     op << "       }\n";
     if (hasArray > 0) {
        op <<"       char *impl_buf=impl_msg->msgBuf+impl_arrstart;\n";
        for(sv=stateVars->begin();!stateVars->end();sv=stateVars->next()) {
           if (sv->arrayLength !=0)
              op << "       memcpy(impl_buf+impl_off_"<<sv->name->charstar()<<
	                 ","<<sv->name->charstar()<<",impl_cnt_"<<sv->name->charstar()<<");\n";
        }  
     }
  op << "       tr->args[" <<paramIndex <<"] = (size_t) impl_msg;\n";
  }   
  int iRef=0, iAny=0;
  for(el=elist->begin(); !elist->end(); el=elist->next()) {
    if(el->con2 == 0) {
      op << "       tr->anyEntries[" << iAny++ << "] = " <<
            el->entryPtr->entryNum << ";\n";
    } else {
      op << "       tr->entries[" << iRef << "] = " << 
            el->entryPtr->entryNum << ";\n";
      op << "       tr->refnums[" << iRef++ << "] = " <<
            el->con2->text->charstar() << ";\n";
    }
  }
  op << "       __cDep->Register(tr);\n";
  op << "       return 0;\n";
  op << "    }\n";

  // end actual code
  op << "  }\n\n";
  // end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else 
   op << "    " << next->label->charstar() << "_end(";

  generateCall(op, *stateVars); 
  
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateWhile(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << constructs->front()->label->charstar() << 
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " <<  next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateFor(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << con1->text->charstar() << ";\n";
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << con3->text->charstar() << ";\n";
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
   generateCall(op, *stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateIf(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if (con2 != 0) {
    op << "      " << con2->label->charstar() << "(";
    generateCall(op, *stateVarsChildren);
    op << ");\n";
  } else {
    op << "      " << label->charstar() << "_end(";
    generateCall(op, *stateVarsChildren);
    op << ");\n";
  }
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateElse(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << constructs->front()->label->charstar() << 
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateForall(XStr& op)
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
  op << "      " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    " << counter->charstar() << "->decrement();\n";
  op << "    if (" << counter->charstar() << "->isDone()) {\n";
  op << "      delete " << counter->charstar() << ";\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "    }\n  }\n\n";
}

void CParseNode::generateOlist(XStr& op)
{
  CParseNode *cn;
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    CCounter *" << counter->charstar() << "= new CCounter(" <<
        constructs->length() << ");\n";
  for(cn=constructs->begin(); 
                     !constructs->end(); cn=constructs->next()) {
    op << "    " << cn->label->charstar() << "(";
    generateCall(op, *stateVarsChildren);
    op << ");\n";
  }
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    " << counter->charstar() << "->decrement();\n";
  op << "    if (" << counter->charstar() << "->isDone()) {\n";
  op << "      delete " << counter->charstar() << ";\n";

  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "    }\n";
  op << "  }\n";
}

void CParseNode::generateOverlap(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else
   op << "    " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateSlist(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, *stateVarsChildren); 
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else
   op << "    " << next->label->charstar() << "_end(";
  //generateCall(op, *stateVars);
  generateCall(op, *stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateSdagEntry(XStr& op)
{
  // header file
  op << "public:\n";
  op << "  void " << con1->text->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << constructs->front()->label->charstar() <<
        "(";
  generateCall(op, *stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
  op << "private:\n";
  op << "  void " << con1->text->charstar() << "_end(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  op << "  }\n\n";
}

void CParseNode::generateAtomic(XStr& op)
{ 
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, *stateVars);
  op << ") {\n";
  op << "    " << text->charstar() << "\n";
  if(nextBeginOrEnd == 1)
    op << "    " << next->label->charstar() << "(";
  else
    op << "    " << next->label->charstar() << "_end(";
  generateCall(op, *stateVars);
  op << ");\n";
  op << "  }\n\n";
}

void CParseNode::generatePrototype(XStr& op, TList<CStateVar*>& list)
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
      if (sv->isconst != 0) 
         op <<sv->isconst->charstar() <<" ";
      if (sv->type1 != 0) 
         op <<sv->type1->charstar() <<" ";
      if (sv->type2 != 0) 
         op <<sv->type2->charstar() <<" ";
      if (sv->byRef != 0)
         op <<" &";
      if (sv->arrayLength != 0) 
         op <<"*";
      else if (sv->allPtrs != 0) 
         op <<sv->allPtrs->charstar() <<" ";
      if (sv->name != 0)
         op <<sv->name->charstar();
    }
    if (sv->isVoid != 1)
       count++;
    sv = list.next();
  }
}

void CParseNode::generateCall(XStr& op, TList<CStateVar*>& list) {
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
}

// boe = 1, if the next call is to begin construct
// boe = 0, if the next call is to end a contruct
void CParseNode::setNext(CParseNode *n, int boe)
{
  switch(type) {
    case SLIST:
      next = n;
      nextBeginOrEnd = boe;
      {
        CParseNode *cn=constructs->begin();
        if (cn==0) // empty slist
          return;
        CParseNode *nextNode=constructs->next();
        for(; nextNode != 0;) {
          cn->setNext(nextNode, 1);
          cn = nextNode;
          nextNode = constructs->next();
        }
        cn->setNext(this, 0);
      }
      return;
    case SDAGENTRY:
    case OVERLAP:
    case OLIST:
    case FORALL:
    case WHEN:
    case FOR:
    case WHILE:
    case ATOMIC:
    case FORWARD:
    case ELSE:
      next = n;
      nextBeginOrEnd = boe;
      n = this; boe = 0; break;
    case IF:
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
  CParseNode *cn;
  if (constructs != 0) {
    for(cn=constructs->begin(); !constructs->end(); cn=constructs->next()) {
      cn->setNext(n, boe);
    }
  }
}

