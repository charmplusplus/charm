#include "CEntry.h"
#include "xi-symbol.h"
#include "CStateVar.h"

using std::list;

namespace xi {

void CEntry::generateDeps(XStr& op)
{
  for(list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end(); ++cn) {
    op << "    __cDep->addDepends(" << (*cn)->nodeNum << "," << entryNum << ");\n";
  }
}

void CEntry::generateCode(XStr& decls, XStr& defs)
{
  CStateVar *sv;
  int i;
  int isVoid = 1;
  int lastWasVoid;
  i = 0;
  decls << "  void ";

  templateGuardBegin(false, defs);
  defs << "void " << decl_entry->getContainer()->baseName() << "::";

  XStr signature;
  signature <<  *entry << "(";
  for(list<CStateVar*>::iterator it = myParameters.begin();
      it != myParameters.end(); ++it, ++i) {
    sv = *it;
    isVoid = sv->isVoid;
    if ((sv->isMsg != 1) && (sv->isVoid != 1)) {
       if (i >0)
         signature <<", ";
       signature << sv->type << " ";
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
         signature << sv->name;
    }
    else if (sv->isVoid != 1){
      if (i < 1) 
         signature << sv->type << " " << sv->name << "_msg";
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
  i = 0;
  if (isVoid == 1) {
     defs << "    __cDep->bufferMessage("<<entryNum<<", (void *) CkAllocSysMsg(), (void*) _bgParentLog, 0);\n";
     defs << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
  }
  else {
     for(list<CStateVar*>::iterator it = myParameters.begin();
         it != myParameters.end(); ++it, ++i) {
       sv = *it;
        if ((i==0) && (sv->isMsg !=1)) {
           defs <<"    int impl_off=0; int impl_arrstart=0;\n";
  	   paramMarshalling = 1;
        }
        if(sv->arrayLength != 0) {
           hasArrays++ ;
	   if (sv->numPtrs > 0)
              printf("ERROR: can't pass pointers across processors \n -- Indicate the array length with []'s, or pass a reference\n");
           defs << "    int impl_off_" << sv->name << ", impl_cnt_" << sv->name << ";\n";
           defs << "    impl_off_" << sv->name << "=impl_off=CK_ALIGN(impl_off,sizeof(" << sv->type << "));\n";
           defs << "    impl_off+=(impl_cnt_" << sv->name << "=sizeof(" << sv->type << ")*(" << sv->arrayLength << "));\n";
        }
        if (paramMarshalling ==0) {
	   defs << "    CmiReference(UsrToEnv(" << sv->name << "_msg));\n";
           if(refNumNeeded) {
              defs << "    int refnum = CkGetRefNum(" <<sv->name << "_msg);\n";
              defs << "    cmsgbuf = __cDep->bufferMessage(" << entryNum << ",(void *) " << sv->name << "_msg , (void *) _bgParentLog, refnum);\n";
              defs << "    tr = __cDep->getTrigger(" << entryNum<<", refnum);\n";
           } else {
              defs << "    cmsgbuf = __cDep->bufferMessage(" << entryNum << ", (void *) " << sv->name << "_msg,  (void *) _bgParentLog, 0);\n";
              defs << "    tr = __cDep->getTrigger(" << entryNum<<", 0);\n";
           } 
        }
        count++;
     }
   }
   if (paramMarshalling == 1) {
     defs <<"    {\n";
     defs <<"      PUP::sizer implP1;\n";
     i = 0;
 
     for(list<CStateVar*>::iterator it = myParameters.begin();
         it != myParameters.end(); ++it, ++i) {
       sv = *it;
        if(sv->arrayLength != 0)
           defs << "      implP1|impl_off_" << sv->name << ";\n";
        else if(sv->byRef != 0)
	   defs << "      implP1|(" <<sv->type << " &)" <<sv->name << ";\n";
	else   
	   defs << "      implP1|" << sv->name << ";\n";
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
     i = 0;
 
     for(list<CStateVar*>::iterator it = myParameters.begin();
         it != myParameters.end(); ++it, ++i) {
       sv = *it;
        if(sv->arrayLength != 0)
           defs << "      implP1|impl_off_" << sv->name << ";\n";
        else if(sv->byRef != 0)
           defs << "      implP1|(" << sv->type << " &)" << sv->name << ";\n";
        else   
	   defs << "      implP1|" << sv->name << ";\n";
     }
     defs <<"    }\n";
     if (hasArrays > 0)
     { //Marshall each array
       defs <<"    char *impl_buf1=impl_msg1->msgBuf+impl_arrstart;\n";
       i = 0;
     for(list<CStateVar*>::iterator it = myParameters.begin();
         it != myParameters.end(); ++it, ++i) {
       sv = *it;
         if(sv->arrayLength != 0) {
           defs << "    memcpy(impl_buf1+impl_off_" << sv->name << "," << sv->name << ",impl_cnt_" << sv->name << ");\n";
	 }
       }
     }
     
     // When a reference number is needed and there are parameters
     // that need marshalling (in other words the parameters of the
     // entry method are not messages) then the first parameter of the
     // entry method is an integer that specifies the reference number
     const char* refNumArg = refNumNeeded ? (*myParameters.begin())->name->charstar() : "0";

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

  if(whenList.size() == 1) {
    defs << "    {\n";
    (*whenList.begin())->generateWhenCode(defs);
    defs << "    }\n";
  }
  else {   
    defs << "    switch(tr->whenID) {\n";
    for(list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end(); ++cn)
    {
      defs << "      case " << (*cn)->nodeNum << ":\n";
      defs << "      {\n";
      // This emits a `return;', so no `break' is needed
      (*cn)->generateWhenCode(defs);
      defs << "      }\n";
    }
    defs << "    }\n";
  } 

  // actual code ends
  defs << "}\n\n";
  templateGuardEnd(defs);
}

}
