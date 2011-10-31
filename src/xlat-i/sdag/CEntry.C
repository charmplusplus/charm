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

void CEntry::generateCode(XStr& op)
{
  CStateVar *sv;
  int i;
  int isVoid = 1;
  int lastWasVoid;
  sv = (CStateVar *)myParameters->begin();
  i = 0;
  op << "  void " << *entry <<"(";
  for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
    isVoid = sv->isVoid;
    if ((sv->isMsg != 1) && (sv->isVoid != 1)) {
       if (i >0)
         op <<", ";
       op << sv->type->charstar() << " ";
       if (sv->arrayLength != 0)
         op << "*";
       else if (sv->byRef != 0) {
         op <<"&";
       }
       if (sv->numPtrs != 0) {
         for(int k = 0; k< sv->numPtrs; k++)
	    op<<"*";
       }
       if (sv->name != 0)
         op << sv->name->charstar();
    }
    else if (sv->isVoid != 1){
      if (i < 1) 
         op << sv->type->charstar() <<" *"<<sv->name->charstar() <<"_msg";
      else
         printf("ERROR: A message must be the only parameter in an entry function\n");
    }
    else
      op <<"void";
  }
  op <<  ") {\n";
  op << "    CWhenTrigger *tr;\n";
  op<<  "    void* _bgParentLog = NULL;\n";
#if CMK_BIGSIM_CHARM
  op<<  "    CkElapse(0.01e-6);\n";
  SdagConstruct::generateTlineEndCall(op);
#endif

  op << "    CMsgBuffer* cmsgbuf;\n";

  int hasArrays = 0;
  int paramMarshalling = 0;
  int count = 0;
  sv = (CStateVar *)myParameters->begin();
  i = 0;
  if (isVoid == 1) {
     op << "   __cDep->bufferMessage("<<entryNum<<", (void *) CkAllocSysMsg(), (void*) _bgParentLog, 0);\n";
     op << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
  }
  else {
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if ((i==0) && (sv->isMsg !=1)) {
           op <<"    int impl_off=0; int impl_arrstart=0;\n";
  	   paramMarshalling = 1;
        }
        if(sv->arrayLength != 0) {
           hasArrays++ ;
	   if (sv->numPtrs > 0)
              printf("ERROR: can't pass pointers across processors \n -- Indicate the array length with []'s, or pass a reference\n");
           op <<"    int impl_off_"<<sv->name->charstar()<<", impl_cnt_"<<sv->name->charstar()<<";\n";
           op <<"    impl_off_"<<sv->name->charstar()<<"=impl_off=CK_ALIGN(impl_off,sizeof("<<sv->type->charstar()<<"));\n";
           op <<"    impl_off+=(impl_cnt_"<<sv->name->charstar()<<"=sizeof("<<sv->type->charstar()<<")*("<<sv->arrayLength->charstar()<<"));\n";
        }
        if (paramMarshalling ==0) {
	   op << "    CmiReference(UsrToEnv(" << sv->name->charstar() << "_msg));\n";
           if(refNumNeeded) {
              op << "    int refnum = CkGetRefNum(" <<sv->name->charstar() <<"_msg);\n";
              op << "    cmsgbuf = __cDep->bufferMessage("<<entryNum<<",(void *) "<<sv->name->charstar() <<"_msg , (void *) _bgParentLog, refnum);\n";
              op << "    tr = __cDep->getTrigger("<<entryNum<<", refnum);\n";
           } else {
              op << "    cmsgbuf = __cDep->bufferMessage("<<entryNum<<", (void *) "<<sv->name->charstar() <<"_msg,  (void *) _bgParentLog, 0);\n";
              op << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
           } 
        }
        count++;
     }
   }
   if (paramMarshalling == 1) {
     op <<"    {\n";
     op <<"      PUP::sizer implP1;\n";
     sv = (CStateVar *)myParameters->begin();
     i = 0;
 
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if(sv->arrayLength != 0)
           op <<"      implP1|impl_off_"<<sv->name->charstar()<<";\n";
        else if(sv->byRef != 0)
	   op <<"      implP1|(" <<sv->type->charstar() <<" &)" <<sv->name->charstar() <<";\n";
	else   
	   op <<"      implP1|"<<sv->name->charstar()<<";\n";
     }
 
     if (hasArrays > 0)
     { //round up pup'd data length--that's the first array
        op <<"      impl_arrstart=CK_ALIGN(implP1.size(),16);\n";
        op <<"      impl_off+=impl_arrstart;\n";
     }
     else  //No arrays--no padding
        op <<"      impl_off+=implP1.size();\n";
  
     op <<"    }\n";

     //Now that we know the size, allocate the packing buffer
     op <<"    CkMarshallMsg *impl_msg1=CkAllocateMarshallMsg(impl_off,NULL);\n";
     //Second pass: write the data
     op <<"    {\n";
     op <<"      PUP::toMem implP1((void *)impl_msg1->msgBuf);\n";
     sv = (CStateVar *)myParameters->begin();
     i = 0;
 
     for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
        if(sv->arrayLength != 0)
           op <<"      implP1|impl_off_"<<sv->name->charstar()<<";\n";
        else if(sv->byRef != 0)
           op <<"      implP1|(" <<sv->type->charstar() <<" &)" <<sv->name->charstar() <<";\n";
        else   
	   op <<"      implP1|"<<sv->name->charstar()<<";\n";
     }
     op <<"    }\n";
     if (hasArrays > 0)
     { //Marshall each array
       op <<"    char *impl_buf1=impl_msg1->msgBuf+impl_arrstart;\n";
       sv = (CStateVar *)myParameters->begin();
       i = 0;
       for(; i<(myParameters->length());i++, sv=(CStateVar *)myParameters->next()) {
         if(sv->arrayLength != 0) {
           op <<"    memcpy(impl_buf1+impl_off_"<<sv->name->charstar()<<","<<sv->name->charstar()<<",impl_cnt_"<<sv->name->charstar()<<");\n";
	 }
       }
     }
     
     if(refNumNeeded) {
     // When a reference number is needed and there are parameters that need marshalling 
     // (in other words the parameters of the entry method are not messages) 
     // then the first parameter of the entry method is an integer that specifies the 
     // reference number
          sv = (CStateVar *)myParameters->begin();
          op << "   cmsgbuf = __cDep->bufferMessage("<<entryNum<<",(void *) impl_msg1, (void*) _bgParentLog,"<<sv->name->charstar()<<");\n";
          op << "    tr = __cDep->getTrigger("<<entryNum<<","<<sv->name->charstar()<<");\n"; 
     } else {
       op << "    cmsgbuf = __cDep->bufferMessage("<<entryNum<<", (void *) impl_msg1, (void*) _bgParentLog, 0);\n";
       op << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
     }
   }

  op << "    if (tr == 0)\n";
  op << "      return;\n"; 

  SdagConstruct::generateTraceEndCall(op);
#if CMK_BIGSIM_CHARM
  SdagConstruct::generateEndExec(op);
#endif

  XStr *whenParams; 
  int iArgs = 0;
  if(whenList.length() == 1) {
    SdagConstruct *cn;
    cn = whenList.begin();
    whenParams = new XStr("");
    sv = (CStateVar *)cn->stateVars->begin();
    i = 0; iArgs = 0;
    lastWasVoid = 0;
    sv = (CStateVar *)cn->stateVars->begin();
    i = 0;
    paramMarshalling = 0;
    lastWasVoid = 0;

#if CMK_BIGSIM_CHARM
    op <<"    cmsgbuf->bgLog2 = (void*)tr->args[1];\n";
    //op << "    " << cn->label->charstar() << "(";    
#endif

    for( i=0; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
      if ((sv->isMsg == 0) && (paramMarshalling == 0) && (sv->isVoid ==0)){
          paramMarshalling =1;
          op << "    CkMarshallMsg *impl_msg = (CkMarshallMsg *) tr->args[" <<iArgs <<"];\n"; 
          op << "    char *impl_buf=((CkMarshallMsg *)impl_msg)->msgBuf;\n";
          op << "    PUP::fromMem implP(impl_buf);\n";
          iArgs++;
      }
      if (sv->isMsg == 1) {
	  if((i!=0) && (lastWasVoid == 0))
	     whenParams->append(", ");
#if CMK_BIGSIM_CHARM
	  if(i==1){   
	    whenParams->append("NULL");
	    lastWasVoid=0;
	    iArgs++;
	    continue;
	  }
#endif
	  whenParams->append("(");
	  whenParams->append(sv->type->charstar());
	  whenParams->append(" *) tr->args[");
	  *whenParams<<iArgs;
	  whenParams->append("]");
	  iArgs++;
       }
       else if (sv->isVoid == 1) 
           // op <<"    CkFreeSysMsg((void  *)tr->args[" <<iArgs++ <<"]);\n";
           op <<"    tr->args[" <<iArgs++ <<"] = 0;\n";
       else if ((sv->isMsg == 0) && (sv->isVoid == 0)) {
          if((i > 0) &&(lastWasVoid == 0)) 
	     whenParams->append(", ");
          whenParams->append(*(sv->name));
          if (sv->arrayLength != 0) 
             op<<"    int impl_off_"<<sv->name->charstar()<<"; implP|impl_off_"<<sv->name->charstar()<<";\n";
          else
             op<<"    "<<sv->type->charstar()<<" "<<sv->name->charstar()<<"; implP|"<<sv->name->charstar()<<";\n";
       }
       lastWasVoid = sv->isVoid;
      
    } 
    if (paramMarshalling == 1) 
        op<<"     impl_buf+=CK_ALIGN(implP.size(),16);\n";
    i = 0;
    sv = (CStateVar *)cn->stateVars->begin();
    for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
       if (sv->arrayLength != 0) 
          op<<"    "<<sv->type->charstar()<<" *"<<sv->name->charstar()<<"=("<<sv->type->charstar()<<" *)(impl_buf+impl_off_"<<sv->name->charstar()<<");\n";
    }

    if (paramMarshalling == 1) 
       op << "    delete (CkMarshallMsg *)impl_msg;\n";
    op << "    " << cn->label->charstar() << "(" << whenParams->charstar();
    op << ");\n";
  
    op << "    delete tr;\n";
#if CMK_BIGSIM_CHARM
    cn->generateTlineEndCall(op);
    cn->generateBeginExec(op, "sdagholder");
#endif
    cn->generateDummyBeginExecute(op);
    op << "    return;\n";
  }
  else {   
    op << "    switch(tr->whenID) {\n";
    for(SdagConstruct *cn=whenList.begin(); !whenList.end(); cn=whenList.next())
    {
      whenParams = new XStr("");
      i = 0; iArgs = 0;
      op << "      case " << cn->nodeNum << ":\n";
      op << "      {\n";
      
#if CMK_BIGSIM_CHARM
      // bgLog2 stores the parent dependence of when, e.g. for, olist
      op <<"  cmsgbuf->bgLog2 = (void*)tr->args[1];\n";	 
#endif
      sv = (CStateVar *)cn->stateVars->begin();
      i = 0;
      paramMarshalling = 0;
      lastWasVoid = 0;

      for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
         if ((sv->isMsg == 0) && (paramMarshalling == 0) && (sv->isVoid ==0)){
            paramMarshalling =1;
            op << "        CkMarshallMsg *impl_msg" <<cn->nodeNum <<" = (CkMarshallMsg *) tr->args["<<iArgs++<<"];\n"; 
            op << "        char *impl_buf" <<cn->nodeNum <<"=((CkMarshallMsg *)impl_msg" <<cn->nodeNum <<")->msgBuf;\n";
            op << "        PUP::fromMem implP" <<cn->nodeNum <<"(impl_buf" <<cn->nodeNum <<");\n";
         }
         if (sv->isMsg == 1) {
	    if((i!=0) && (lastWasVoid == 0))
	      whenParams->append(", ");
#if CMK_BIGSIM_CHARM
	    if(i==1) {
	       whenParams->append(" NULL ");
	       lastWasVoid=0;
	       // skip this arg which is supposed to be _bgParentLog
	       iArgs++;
	       continue;
	    }
#endif
	    whenParams->append("(");
	    whenParams->append(sv->type->charstar());
	    whenParams->append(" *) tr->args[");
	    *whenParams<<iArgs;
	    whenParams->append("]");
	    iArgs++;
         }
         else if (sv->isVoid == 1) 
            // op <<"    CkFreeSysMsg((void  *)tr->args[" <<iArgs++ <<"]);\n";
            op <<"    tr->args[" <<iArgs++ <<"] = 0;\n";
         else if ((sv->isMsg == 0) && (sv->isVoid == 0)) {
            if((i > 0) && (lastWasVoid == 0))
	       whenParams->append(", ");
            whenParams->append(*(sv->name));
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
      op << "        " << cn->label->charstar() << "(" << whenParams->charstar();
      op << ");\n";
      op << "        delete tr;\n";

#if CMK_BIGSIM_CHARM
      cn->generateTlineEndCall(op);
      cn->generateBeginExec(op, "sdagholder");
#endif
      cn->generateDummyBeginExecute(op);

      op << "        return;\n";
      op << "      }\n";
    }
  op << "    }\n";
  } 

  // actual code ends
  op << "  }\n\n";
}

}
