#include "CEntry.h"
#include "xi-symbol.h"
#include "CStateVar.h"

namespace xi {

void CEntry::generateDeps(XStr& op)
{
  SdagConstruct *cn;
  for(cn=whenList.begin(); !whenList.end(); cn=whenList.next()) {
    op << "    __cDep->addDepends("<<cn->nodeNum<<","<<entryNum<<");\n";
  }
}

static void generateWhenCode(XStr& op, SdagConstruct *cn)
{
  XStr whenParams = "";
  CStateVar *sv = cn->stateVars->begin();
  int i = 0;
  int iArgs = 0;
  bool lastWasVoid = false;
  bool paramMarshalling = false;

#if CMK_BIGSIM_CHARM
  // bgLog2 stores the parent dependence of when, e.g. for, olist
  op <<"  cmsgbuf->bgLog2 = (void*)tr->args[1];\n";
#endif

  for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
    if ((sv->isMsg == 0) && (paramMarshalling == 0) && (sv->isVoid ==0)){
      paramMarshalling =1;
      op << "        CkMarshallMsg *impl_msg" <<cn->nodeNum <<" = (CkMarshallMsg *) tr->args["<<iArgs++<<"];\n";
      op << "        char *impl_buf" <<cn->nodeNum <<"=((CkMarshallMsg *)impl_msg" <<cn->nodeNum <<")->msgBuf;\n";
      op << "        PUP::fromMem implP" <<cn->nodeNum <<"(impl_buf" <<cn->nodeNum <<");\n";
    }
    if (sv->isMsg == 1) {
      if((i!=0) && (lastWasVoid == 0))
        whenParams.append(", ");
#if CMK_BIGSIM_CHARM
      if(i==1) {
        whenParams.append(" NULL ");
        lastWasVoid=0;
        // skip this arg which is supposed to be _bgParentLog
        iArgs++;
        continue;
      }
#endif
      whenParams.append("(");
      whenParams.append(sv->type->charstar());
      whenParams.append(") tr->args[");
      whenParams<<iArgs;
      whenParams.append("]");
      iArgs++;
    }
    else if (sv->isVoid == 1)
      // op <<"    CkFreeSysMsg((void  *)tr->args[" <<iArgs++ <<"]);\n";
      op <<"        tr->args[" <<iArgs++ <<"] = 0;\n";
    else if ((sv->isMsg == 0) && (sv->isVoid == 0)) {
      if((i > 0) && (lastWasVoid == 0))
        whenParams.append(", ");
      whenParams.append(*(sv->name));
      if (sv->arrayLength != 0)
        op<<"        int impl_off"<<cn->nodeNum <<"_"<<sv->name->charstar()<<"; implP"
          <<cn->nodeNum <<"|impl_off" <<cn->nodeNum <<"_"<<sv->name->charstar()<<";\n";
      else
        op<<"        "<<sv->type->charstar()<<" "<<sv->name->charstar()<<"; implP"
          <<cn->nodeNum <<"|"<<sv->name->charstar()<<";\n";
    }
    lastWasVoid = sv->isVoid;
  }
  if (paramMarshalling == 1)
    op<<"        impl_buf"<<cn->nodeNum <<"+=CK_ALIGN(implP" <<cn->nodeNum <<".size(),16);\n";
  i = 0;
  sv = (CStateVar *)cn->stateVars->begin();
  for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
    if (sv->arrayLength != 0)
      op<<"        "<<sv->type->charstar()<<" *"<<sv->name->charstar()<<"=("<<sv->type->charstar()<<" *)(impl_buf" <<cn->nodeNum
        <<"+impl_off" <<cn->nodeNum <<"_"<<sv->name->charstar()<<");\n";
  }
  if (paramMarshalling == 1)
    op << "        delete (CkMarshallMsg *)impl_msg" <<cn->nodeNum <<";\n";
  op << "        " << cn->label->charstar() << "(" << whenParams.charstar();
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

void CEntry::generateCode(XStr& decls, XStr& defs)
{
  CStateVar *sv;
  int i;
  int isVoid = 1;
  int lastWasVoid;
  sv = (CStateVar *)myParameters->begin();
  i = 0;
  decls << "  void ";

  templateGuardBegin(false, defs);
  defs << "void " << decl_entry->getContainer()->baseName() << "::";

  XStr signature;
  signature <<  *entry << "(";
  for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
    isVoid = sv->isVoid;
    if ((sv->isMsg != 1) && (sv->isVoid != 1)) {
       if (i >0)
         signature <<", ";
       signature << sv->type->charstar() << " ";
       if (sv->arrayLength != 0)
         signature << "*";
       else if (sv->byRef != 0) {
         signature <<"&";
       }
       if (sv->numPtrs != 0) {
         for(int k = 0; k< sv->numPtrs; k++)
	    signature << "*";
       }
       if (sv->name != 0)
         signature << sv->name->charstar();
    }
    else if (sv->isVoid != 1){
      if (i < 1) 
         signature << sv->type->charstar() <<" "<<sv->name->charstar() <<"_msg";
      else
         printf("ERROR: A message must be the only parameter in an entry function\n");
    }
    else
      signature <<"void";
  }
  signature << ")";

  decls << signature << ";\n";

  defs << signature << "{\n";
  defs << "    CWhenTrigger *tr;\n";
  defs << "    void* _bgParentLog = NULL;\n";
#if CMK_BIGSIM_CHARM
  defs <<  "    CkElapse(0.01e-6);\n";
  SdagConstruct::generateTlineEndCall(defs);
#endif

  defs << "    CMsgBuffer* cmsgbuf;\n";

  int hasArrays = 0;
  int paramMarshalling = 0;
  int count = 0;
  sv = (CStateVar *)myParameters->begin();
  i = 0;
  if (isVoid == 1) {
     defs << "    __cDep->bufferMessage("<<entryNum<<", (void *) CkAllocSysMsg(), (void*) _bgParentLog, 0);\n";
     defs << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
  }
  else {
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if ((i==0) && (sv->isMsg !=1)) {
           defs <<"    int impl_off=0; int impl_arrstart=0;\n";
  	   paramMarshalling = 1;
        }
        if(sv->arrayLength != 0) {
           hasArrays++ ;
	   if (sv->numPtrs > 0)
              printf("ERROR: can't pass pointers across processors \n -- Indicate the array length with []'s, or pass a reference\n");
           defs <<"    int impl_off_"<<sv->name->charstar()<<", impl_cnt_"<<sv->name->charstar()<<";\n";
           defs <<"    impl_off_"<<sv->name->charstar()<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type->charstar()<<"));\n";
           defs <<"    impl_off+=(impl_cnt_"<<sv->name->charstar()<<"=sizeof("<<sv->type->charstar()<<")*("<<sv->arrayLength->charstar()<<"));\n";
        }
        if (paramMarshalling ==0) {
	   defs << "    CmiReference(UsrToEnv(" << sv->name->charstar() << "_msg));\n";
           if(refNumNeeded) {
              defs << "    int refnum = CkGetRefNum(" <<sv->name->charstar() <<"_msg);\n";
              defs << "    cmsgbuf = __cDep->bufferMessage("<<entryNum<<",(void *) "<<sv->name->charstar() <<"_msg , (void *) _bgParentLog, refnum);\n";
              defs << "    tr = __cDep->getTrigger("<<entryNum<<", refnum);\n";
           } else {
              defs << "    cmsgbuf = __cDep->bufferMessage("<<entryNum<<", (void *) "<<sv->name->charstar() <<"_msg,  (void *) _bgParentLog, 0);\n";
              defs << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
           } 
        }
        count++;
     }
   }
   if (paramMarshalling == 1) {
     defs <<"    {\n";
     defs <<"      PUP::sizer implP1;\n";
     sv = (CStateVar *)myParameters->begin();
     i = 0;
 
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if(sv->arrayLength != 0)
           defs <<"      implP1|impl_off_"<<sv->name->charstar()<<";\n";
        else if(sv->byRef != 0)
	   defs <<"      implP1|(" <<sv->type->charstar() <<" &)" <<sv->name->charstar() <<";\n";
	else   
	   defs <<"      implP1|"<<sv->name->charstar()<<";\n";
     }
 
     if (hasArrays > 0)
     { //round up pup'd data length--that's the first array
        defs <<"      impl_arrstart=CK_ALIGN(implP1.size(),16);\n";
        defs <<"      impl_off+=impl_arrstart;\n";
     }
     else  //No arrays--no padding
        defs <<"      impl_off+=implP1.size();\n";
  
     defs <<"    }\n";

     //Now that we know the size, allocate the packing buffer
     defs <<"    CkMarshallMsg *impl_msg1=CkAllocateMarshallMsg(impl_off,NULL);\n";
     //Second pass: write the data
     defs <<"    {\n";
     defs <<"      PUP::toMem implP1((void *)impl_msg1->msgBuf);\n";
     sv = (CStateVar *)myParameters->begin();
     i = 0;
 
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if(sv->arrayLength != 0)
           defs <<"      implP1|impl_off_"<<sv->name->charstar()<<";\n";
        else if(sv->byRef != 0)
           defs <<"      implP1|(" <<sv->type->charstar() <<" &)" <<sv->name->charstar() <<";\n";
        else   
	   defs <<"      implP1|"<<sv->name->charstar()<<";\n";
     }
     defs <<"    }\n";
     if (hasArrays > 0)
     { //Marshall each array
       defs <<"    char *impl_buf1=impl_msg1->msgBuf+impl_arrstart;\n";
       sv = (CStateVar *)myParameters->begin();
       i = 0;
       for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
         if(sv->arrayLength != 0) {
           defs <<"    memcpy(impl_buf1+impl_off_"<<sv->name->charstar()<<","<<sv->name->charstar()<<",impl_cnt_"<<sv->name->charstar()<<");\n";
	 }
       }
     }
     
     // When a reference number is needed and there are parameters
     // that need marshalling (in other words the parameters of the
     // entry method are not messages) then the first parameter of the
     // entry method is an integer that specifies the reference number
     const char* refNumArg = refNumNeeded ? myParameters->begin()->name->charstar() : "0";

     defs << "    cmsgbuf = __cDep->bufferMessage(" << entryNum
        << ", (void *) impl_msg1, (void*) _bgParentLog, "
        << refNumArg <<  ");\n";
     defs << "    tr = __cDep->getTrigger(" << entryNum << ", "
        << refNumArg << ");\n";
   }

  defs << "    if (tr == 0)\n";
  defs << "      return;\n";

  SdagConstruct::generateTraceEndCall(defs);
#if CMK_BIGSIM_CHARM
  SdagConstruct::generateEndExec(defs);
#endif

  if(whenList.length() == 1) {
    defs << "    {\n";
    generateWhenCode(defs, whenList.begin());
    defs << "    }\n";
  }
  else {   
    defs << "    switch(tr->whenID) {\n";
    for(SdagConstruct *cn=whenList.begin(); !whenList.end(); cn=whenList.next())
    {
      defs << "      case " << cn->nodeNum << ":\n";
      defs << "      {\n";
      // This emits a `return;', so no `break' is needed
      generateWhenCode(defs, cn);
      defs << "      }\n";
    }
    defs << "    }\n";
  } 

  // actual code ends
  defs << "}\n\n";
  templateGuardEnd(defs);
}

}
