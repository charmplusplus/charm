
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iostream.h>
#include <fstream.h>
#include "xi-symbol.h"
#include "xi-parse.tab.h"

extern Module *thismodule ;

int moduleHasMain = 0;

void GenerateStructsFns(ofstream& top, ofstream& bot) ;
void GenerateRegisterCalls(ofstream& top, ofstream& bot) ;
void GenerateProxies(ofstream& top, ofstream& bot);


void Generate(char *interfacefile)
{
  char modulename[1024], topname[1024], botname[1024], definename[1024] ;
  strcpy(modulename, interfacefile) ;
  modulename[strlen(interfacefile)-3] = '\0' ; // assume ModuleName.ci
  strcpy(topname,modulename) ;
  strcat(topname,".top.h") ;
  strcpy(botname,modulename) ;
  strcat(botname,".bot.h") ;

  ofstream top(topname), bot(botname) ;

  if (top == 0 || bot == 0) {
    cerr << "Cannot open " << topname 
         << " or " << botname << " for writing !!" << endl;
    exit(1);
  }

  sprintf(definename, "CI_%s_TOP_H", thismodule->name);
  top << "#ifndef " << definename << "\n#define " << definename << endl;
  sprintf(definename, "CI_%s_BOT_H", thismodule->name);
  bot << "#ifndef " << definename << "\n#define " << definename << endl;

  GenerateStructsFns(top, bot) ;
  GenerateRegisterCalls(top, bot) ;
  GenerateProxies(top, bot);

  top << "#endif\n";
  bot << "#endif\n";
}


void commonStuff(ofstream& top, ofstream& bot, Chare *c, Entry *e)
{
  char str[2048] ;

  /* This is the constructor EP */
  if ( strcmp(c->name, e->name) == 0 ) {

    if (e->isMessage())
      sprintf(str,"\tnew (obj) %s((%s *)m) ;",
	      c->name, e->msgtype->name) ;
    else
      sprintf(str,"\tnew (obj) %s() ;", c->name);

    bot << str << endl ;

    // ERROR if isReturnMsg()

  } else if (e->isReturnMsg()){ // Returns a message
    bot << "ENVELOPE *env = ENVELOPE_UPTR(m);" << endl;
    bot << "\tint i = GetEnv_ref(env);" << endl;
    bot << "\tint j = GetEnv_pe(env);" << endl;
    sprintf(str, "\t%s *m2 = ((%s *)obj)->%s((%s *)m);",
      e->returnMsg->name, c->name, e->name, e->msgtype->name);
    bot << str << endl ;
    bot << "\tSetRefNumber( (void*) m2, i);" << endl;
    bot << "\tCSendToFuture( (void*) m2, j);" << endl;

  } else { // Regular EP
    sprintf(str,"\t((%s *)obj)->%s((%s *)m) ;",
      c->name,e->name,e->msgtype->name) ;
    bot << str << endl ;
  }
}

void GenerateStructsFns(ofstream& top, ofstream& bot)
{
  char str[2048] ;


  Chare *c; 
  Entry *e;

//  bot << "#ifndef CI_THREAD_WRAPPER\n#define CI_THREAD_WRAPPER\n";
  bot << "typedef struct { void *obj, *m; ";
  bot << "CHARE_BLOCK *chareblock; } Element;" << endl ;
  bot << "CpvExtern(CHARE_BLOCK *,currentChareBlock);" << endl ;
//  bot << "#endif\n";

  sprintf(str,"extern char *_CK_%s_id;", thismodule->name);
  top << str << endl;

  /* Output all chare and EP id variables. Note : if this chare is not
     defined in this module, put "extern" and dont initialize.  */

  for (c=thismodule->chares; c!=NULL; c=c->next ) {

    sprintf(str,"extern int _CK_chare_%s ;",c->name) ;
    top << str << endl ;
    if (!c->isExtern()){
      sprintf(str,"int _CK_chare_%s = _CK_%s_id[0];",c->name, thismodule->name) ;
      bot << str << endl ;
    }

    for (e=c->entries; e!=NULL; e=e->next ) {
      if(e->isMessage()) {
        sprintf(str,"extern int _CK_ep_%s_%s_%s;",c->name,e->name,e->msgtype->name) ;
      } else {
        sprintf(str,"extern int _CK_ep_%s_%s;",c->name,e->name) ;
      }
      top << str << endl ;
      if (!c->isExtern()) {
        if(e->isMessage()) {
          sprintf(str,"int _CK_ep_%s_%s_%s = _CK_%s_id[0] ;",c->name,e->name,e->msgtype->name,thismodule->name) ;
        } else {
          sprintf(str,"int _CK_ep_%s_%s = _CK_%s_id[0] ;",c->name,e->name,thismodule->name) ;
        }
        bot << str << endl ;
      }
    } // endfor e
  }



  /* Output EP stub functions. Note : we assume main::main always
     has argc-argv. */
  for ( c=thismodule->chares; c!=NULL; c=c->next ) {
    for (e=c->entries; e!=NULL; e=e->next ) {

      // If this is the main::main EP
      if ( strcmp(c->name,"main")==0 && 
            strcmp(e->name,"main")==0 ) {
        bot << "extern \"C\" void _CK_call_main_main(void *m, void *obj, ";
	bot << "int argc, char *argv[]);" << endl ;
        bot << "void _CK_call_main_main(void *m, void *obj, ";
	bot << "int argc, char *argv[])" << endl ;
        bot << "{" << endl ;
        bot << "\tnew (obj) main(argc,argv) ;" << endl;
        bot << "}" << endl ;
        moduleHasMain = 1;
        // ERROR if isThreaded() or isReturnMsg()
        continue ;
      }

      // Is this a threaded EP
      if (e->isThreaded()){
        if(e->isMessage()) {
          sprintf(str,"void _CK_call_threaded_%s_%s_%s(void *velt)",
            c->name,e->name,e->msgtype->name);
        } else {
          sprintf(str,"void _CK_call_threaded_%s_%s(void *velt)",
            c->name,e->name);
        }
        bot << str << endl ;
        bot << "{" << endl ;
        bot << "\tElement *elt = (Element *) velt;" << endl;
        bot << "\tvoid *obj = elt->obj;" << endl;
        bot << "\tvoid *m = elt->m;" << endl;
        bot << "\tCpvAccess(currentChareBlock) = elt->chareblock;" << endl;

        commonStuff(top, bot, c, e);

        bot << "\tCmiFree(elt);" << endl;
        bot << "\tCthFree(CthSelf());" << endl;
        bot << "\tCthSuspend();" << endl;
        bot << "}" << endl ;

        if(e->isMessage()) {
          sprintf(str,"void _CK_call_%s_%s_%s(void *m, void *obj)",
            c->name,e->name,e->msgtype->name);
        } else {
          sprintf(str,"void _CK_call_%s_%s(void *m, void *obj)",
            c->name,e->name);
        }
        bot << str << endl ;
        bot << "{" << endl ;
        bot << "\tCthThread t;" << endl;
        bot << "\tElement *element = (Element *) CmiAlloc(sizeof(Element));" ;
	bot << endl;
        bot << "\telement->m = m;\n\telement->obj = obj;" << endl;
        bot << "\telement->chareblock = CpvAccess(currentChareBlock) ;"
            << endl;

        if(e->isMessage()) {
          sprintf(str,
            "\tt = CthCreate( (CthVoidFn) _CK_call_threaded_%s_%s_%s, (void *) element,%d);",
            c->name,e->name,e->msgtype->name,e->get_stackSize()) ;
        } else {
          sprintf(str,
            "\tt = CthCreate( (CthVoidFn) _CK_call_threaded_%s_%s, (void *) element,%d);",
            c->name,e->name,e->get_stackSize()) ;
        }
        bot << str << endl;
        bot << "\tCthSetStrategyDefault(t);" << endl;
        bot << "\tCthAwaken(t);" << endl;

        bot << "}" << endl ;

      } else { // NOT threaded
        if(e->isMessage()) {
          sprintf(str,"void _CK_call_%s_%s_%s(void *m, void *obj)",
            c->name,e->name,e->msgtype->name);
        } else {
          sprintf(str,"void _CK_call_%s_%s(void *m, void *obj)",
            c->name,e->name);
        }
        bot << str << endl ;
        bot << "{" << endl ;

        commonStuff(top, bot, c, e);

        bot << "}" << endl ;
      }

    } // endfor e =
  } // endfor c =


  ReadOnly *r;
  /* Output ids for readonly messages */
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) 
    if ( r->ismsg ) {
      //       top << "int _CK_index_" << r->name << ";" << endl ;
      // this declaration is needed only in bot. changed on 11/20/96 - sanjay
      bot << "int _CK_index_" << r->name << ";" << endl ;
// #### add isExtern to Readonly
//      if (!r->isExtern())
//        bot << "int _CK_index_" << r->name << ";" << endl ;
    }


  Message *m;
  /* Output ids for message types */
  for ( m=thismodule->messages; m!=NULL; m=m->next ) {
    top << "extern int _CK_msg_" << m->name << ";" << endl ;
    if (!m->isExtern())
      bot << "int _CK_msg_" << m->name << "=0;" << endl ;
  }



  /* for allocked MsgTypes output the alloc stub functions */
  for ( m=thismodule->messages; m!=NULL; m=m->next ) {
    sprintf(str,
      "static comm_object *_CK_coerce_%s(void *msg)\n{\n",
      m->name);
    bot << str;
    sprintf(str, "\treturn (comm_object *) new (msg) %s;\n}\n", m->name);
    bot << str;
    if ( !m->allocked )
      continue ;
    sprintf(str,
    "static void *_CK_alloc_%s(int msgno, int size, int *array, int prio)\n{\n",
      m->name) ; 
    bot << str ;
    sprintf(str, "\tvoid *out;\n");
    bot << str;
    sprintf(str,"\tout = %s::alloc(msgno,size,array,prio);\n",m->name) ;
    bot << str ;
    sprintf(str, "\treturn out;\n}\n");
    bot << str ;
  }

  /* for packable MsgTypes output the pack - unpack stub functions */
  for ( m=thismodule->messages; m!=NULL; m=m->next ) {
    if ( !m->packable )
      continue ;

    sprintf(str,
      "static void _CK_pack_%s(void *in, void **out, int *length)\n{\n",
      m->name) ; 
    bot << str ;
    sprintf(str,"\t(*out) = ((%s *)in)->pack(length) ;\n}\n",m->name) ;
    bot << str ;


    sprintf(str,"static void _CK_unpack_%s(void *in, void **out)\n{\n",
	    m->name);
    bot << str ;
    if (!m->allocked)  // Varsize message don't need alloc
    {
      sprintf(str,
      "\t%s * m = (%s *)GenericCkAlloc(_CK_msg_%s,sizeof(%s),0) ;\n",
      m->name, m->name, m->name, m->name) ;
      bot << str ;
      bot << "\t(*out) = (void *) m ;" << endl ;
      bot << "\tm->unpack(in) ;\n}" << endl ;
    } else {
      bot << "\t(*out) = (void *) in ;" << endl ;
      sprintf(str,"\t((%s *)(*out))->unpack(in) ;\n}\n",m->name);
      bot << str;
    }
  }



  /* Output _CK_mod_CopyFromBuffer, _CK_mod_CopyToBuffer for readonlys */
  bot << "extern \"C\" void _CK_13CopyFromBuffer(void *,int) ;" << endl ;
  bot << "void _CK_" << thismodule->name
    << "_CopyFromBuffer(void **_CK_ReadMsgTable)\n{" << endl ;

  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( r->ismsg )
      continue ;
    sprintf(str,"\t_CK_13CopyFromBuffer(&%s,sizeof(%s)) ;\n",
      r->name,r->name) ;
    bot << str ;
  }
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( !r->ismsg )
      continue ;
    sprintf(str,"    {   void **temp = (void **)(&%s);\n",r->name) ;
    bot << str ;
    sprintf(str,"        *temp = _CK_ReadMsgTable[_CK_index_%s];  }\n",
      r->name);
    bot << str ;
  }
  bot << "}" << endl ;


  bot << "extern \"C\" void _CK_13CopyToBuffer(void *,int) ;" << endl ;
  bot << "extern \"C\" void ReadMsgInit(void *,int) ;" << endl ;
  bot << "void _CK_" << thismodule->name << "_CopyToBuffer()" << endl ;
  bot << "{" << endl ;
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( r->ismsg )
      continue ;
    sprintf(str,"\t_CK_13CopyToBuffer(&%s,sizeof(%s)) ;\n",r->name,r->name) ;
    bot << str ;
  }
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( !r->ismsg )
      continue ;
    sprintf(str,"\tReadMsgInit(%s,_CK_index_%s) ;\n",r->name,r->name) ;
    bot << str ;
  }
  bot << "}\n\n\n" ;
}






void GenerateRegisterCalls(ofstream& top, ofstream& bot)
{
  char str[2048] ;

/* generate the beginning of this Module's Init function */
  sprintf(str,"char *_CK_%s_id=\"\\0charmc autoinit %s\";\n",
	  thismodule->name, thismodule->name);
  bot << str ;
  sprintf(str,"extern \"C\" void _CK_%s_init() ;\n",thismodule->name) ;
  bot << str ;
  sprintf(str,"void _CK_%s_init()\n{\n",thismodule->name) ;
  bot << str ;

/* first register all messages */
  for ( Message *m=thismodule->messages; m!=NULL; m=m->next ) {
    if( m->allocked)
      sprintf(str,
        "_CK_msg_%s = registerMsg(\"%s\", (FUNCTION_PTR)&_CK_alloc_%s, ",
        m->name, m->name, m->name) ;
    else
      sprintf(str,
        "_CK_msg_%s = registerMsg(\"%s\", (FUNCTION_PTR)&GenericCkAlloc, ",
        m->name, m->name) ;
    bot << str ;

    if ( !m->packable ) 
      sprintf(str,"0, 0, ") ; 
    else 
      sprintf(str,
        "(FUNCTION_PTR)&_CK_pack_%s, (FUNCTION_PTR)&_CK_unpack_%s, ",
        m->name, m->name) ;
    bot << str ;

    sprintf(str,"(FUNCTION_PTR) &_CK_coerce_%s, sizeof(%s)) ;\n\n",
                m->name, m->name) ;
    bot << str ;
  }
  sprintf(str,"\n\n") ;


/* now register all chares and BOCs and their EPs */
  for ( Chare *chare=thismodule->chares; chare!=NULL; chare=chare->next )
  {
    sprintf(str,"_CK_chare_%s = registerChare(\"%s\", sizeof(%s), 0) ;\n\n",
      chare->name,chare->name,chare->name) ;
    bot << str ;

    for  ( Entry *ep=chare->entries; ep!=NULL; ep=ep->next ) {

      if ( chare->chareboc == CHARE ) {
        if(ep->isMessage()) {
          sprintf(str,
           "_CK_ep_%s_%s_%s = registerEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s_%s, 1,",
           chare->name, ep->name, ep->msgtype->name, ep->name, chare->name,ep->name,ep->msgtype->name) ;
        } else {
          sprintf(str,
           "_CK_ep_%s_%s = registerEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, 1,",
           chare->name, ep->name, ep->name, chare->name,ep->name) ;
        }
      } else {
       if(ep->isMessage()) {
         sprintf(str,
         "_CK_ep_%s_%s_%s = registerBocEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s_%s, 1,",
           chare->name, ep->name, ep->msgtype->name, ep->name, chare->name,ep->name, ep->msgtype->name) ;
       } else {
         sprintf(str,
         "_CK_ep_%s_%s = registerBocEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, 1,",
           chare->name, ep->name, ep->name, chare->name,ep->name) ;
       }
      }

      bot << str ;

      if ( strcmp(chare->name,"main")==0 && strcmp(ep->name,"main")==0 ) 
        sprintf(str,"0, _CK_chare_%s) ;\n\n",chare->name) ;
      else if (ep->isMessage())
        sprintf(str,"_CK_msg_%s, _CK_chare_%s) ;\n\n",
          ep->msgtype->name, chare->name) ;
      else
        sprintf(str,"0, _CK_chare_%s) ;\n\n", chare->name) ;
      bot << str ;

//if (ep->isThreaded()){
//  sprintf(str, "setThreadedEp(_CK_ep_%s_%s);\n\n", chare->name, ep->name) ;
//  bot << str ;
//}

    } // end for ep =
  } // end for chare =
  bot << "\n\n" ;


if (moduleHasMain)
{
/* register the main chare */
  sprintf(str,
    "registerMainChare(_CK_chare_main, _CK_ep_main_main, 1) ;\n\n\n");
  bot << str ;
}


/* register distributed-table-variables */
  for ( Table *t=thismodule->tables; t!=NULL; t=t->next ) {
    sprintf(str,"%s.SetId(registerTable(\"%s\", 0, 0)) ;\n", t->name, t->name);
    bot << str ;
  }
  bot << "\n\n" ;


/* now register readonlies and readonli messages */
  sprintf(str,"int readonlysize=0 ;\n") ;
  bot << str ;
  ReadOnly *r;
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( r->ismsg )
      continue ;
    sprintf(str,"readonlysize += sizeof(%s) ;\n",r->name) ;
    bot << str ;
  }

  sprintf(str,
    "registerReadOnly(readonlysize, (FUNCTION_PTR)&_CK_%s_CopyFromBuffer, (FUNCTION_PTR)&_CK_%s_CopyToBuffer) ;\n",
    thismodule->name,thismodule->name) ;
  bot << str ;


  /* this is only for giving a unique index to all all readonly msgs */
  for ( r=thismodule->readonlys; r!=NULL; r=r->next ) {
    if ( !r->ismsg )
      continue ;
    sprintf(str,"_CK_index_%s = registerReadOnlyMsg() ;\n",r->name) ;
    bot << str ;
  }


/* This is the closing brace of the Module-init function */
  bot << "\n}\n" ;
}

static void
spew(ofstream &strm, const char *ostr,
     char *a1 = "ERROR", char *a2 = "ERROR",
     char *a3 = "ERROR", char *a4 = "ERROR",
     char *a5 = "ERROR")
{
  int i;
  for(i=0; i<strlen(ostr); i++){
    switch(ostr[i]){
    case '\01':
      strm << a1; break;
    case '\02':
      strm << a2; break;
    case '\03':
      strm << a3; break;
    case '\04':
      strm << a4; break;
    case '\05':
      strm << a5; break;
    default:
      strm << ostr[i];
    }
  }
}

const char *CImessage = // msgType
"class \01;\n"
"class CMessage_\01 : public comm_object {\n"
"  public:\n"
"    void *operator new(CMK_SIZE_T size, void *ptr) { return ptr; }\n"
"    void *operator new(CMK_SIZE_T size) {\n"
"      return (void *)GenericCkAlloc(MsgIndex(\01), size, 0) ;\n"
"    };\n"
"    void *operator new(CMK_SIZE_T size, int priobits) {\n"
"      return (void *)GenericCkAlloc(MsgIndex(\01), size, priobits) ;\n"
"    };\n"
"    void *operator new(CMK_SIZE_T size, int *sizes) {\n"
"      return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[MsgIndex(\01)].alloc))(MsgIndex(\01), size, sizes, 0);\n"
"    };\n"
"    void *operator new(CMK_SIZE_T size, int priobits, int *sizes) {\n"
"      return (void *)((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[MsgIndex(\01)].alloc))(MsgIndex(\01), size, sizes, priobits);\n"
"    };\n"
"};\n"
;

const char *CIAsyncChareCreateProto = // charename, msgname
"extern void CAsync_\01(\02 *m, int onPE=CK_PE_ANY);\n"
;

const char *CIAsyncChareCreateImpl = // charename, msgname
"void CAsync_\01(\02 *m, int onPE) {\n"
"  new_chare2(\01, \02, m, (ChareIDType *)0, onPE);\n"
"};\n"
;

const char *CIAsyncGroupCreateProto = // groupname, msgname
"extern int CAsync_\01(\02 *m);\n"
;

const char *CIAsyncGroupCreateImpl = // groupname, msgname
"int CAsync_\01(\02 *m) {\n"
"  return new_group(\01, \02, m);\n"
"};\n"
;

const char *CIProxyClassStart = // classname
"class \01;\n"
"class CProxy_\01 {\n"
"  private:\n"
"    ChareIDType cid;\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(ChareIDType _cid) { cid = _cid; };\n"
;

const char *CIProxyClassConstructor = // classname, msgname
"    CProxy_\01(\02 *m, int pe=CK_PE_ANY) {\n"
"      new_chare2(\01, \02, m, &cid, pe);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(\01,\01,\02);\n"
"    };\n"
;

const char *CIProxyClassMethod = // classname, methodname, msgname
"    void \02(\03 *m) {\n"
"      CSendMsg(\01, \02, \03, m, &cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

const char *CIProxyClassRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m) {\n"
"      return (\04 *) CRemoteCallFn(GetEntryPtr(\01,\02,\03), m, &cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

const char *CIProxyClassEnd =
"    ChareIDType _getCid(void) { return cid; }; \n"
"    void _setCid(ChareIDType _cid) { cid = _cid; }; \n"
"};\n"
;

const char *CIProxyMainStart =
"class main;\n"
"class CProxy_main {\n"
"  private:\n"
"  int dummy;\n"
"  public:\n"
"    CProxy_main() {};\n"
;

const char *CIProxyMainMethod = // methodname, msgname
"    void \01(\02 *m) {\n"
"      CSendMsg(main, \01, \02, m, &mainhandle);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(main,\01,\02);\n"
"    };\n"
;

const char *CIProxyMainRetMethod = // methodname, msgname, retmsg
"    \03 *\01(\02 *m) {\n"
"      return (\03 *) CRemoteCallFn(GetEntryPtr(main,\01,\02), m, &mainhandle);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(main,\01,\02);\n"
"    };\n"
;

const char *CIProxyMainEnd =
"};\n"
"extern CProxy_main mainproxy;\n"
;

const char *CIProxyMainDef =
"CProxy_main mainproxy;\n"
;

const char *CIProxyGroupStart = // classname
"class \01;\n"
"class CProxy_\01 {\n"
"  private:\n"
"    int bocid;\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(int _bocid) { bocid = _bocid; };\n"
;

const char *CIProxyGroupConstructor = // classname, msgname
"    CProxy_\01(\02 *m) {\n"
"      bocid = new_group(\01, \02, m);\n"
"    };\n"
"    CProxy_\01(\02 *m, int retEP, ChareIDType *retID) {\n"
"      new_group2(\01, \02, m, retEP, retID);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(\01,\01,\02);\n"
"    };\n"
;

const char *CIProxyGroupMethod = // classname, methodname, msgname
"    void \02(\03 *m) {\n"
"      CBroadcastMsgBranch(\01, \02, \03, m, bocid);\n"
"    };\n"
"    void \02(\03 *m, int onPE) {\n"
"      CSendMsgBranch(\01, \02, \03, m, bocid, onPE);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

const char *CIProxyGroupRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m, int onPE) {\n"
"      return (\04 *) CRemoteCallBranchFn(GetEntryPtr(\01,\02,\03), m, bocid, onPE);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

const char *CIProxyGroupEnd = // classname
"    int _getBocid(void) { return bocid; }; \n"
"    void _setBocid(int _bocid) { bocid = _bocid; }; \n"
"    \01 *_localBranch(void) {\n"
"      return (\01 *) CLocalBranch(\01, bocid);\n"
"    };\n"
"};\n"
;

void GenerateProxies(ofstream& top, ofstream& bot)
{
  char str[2048];

  Message *m;
  Chare *c;
  Entry *e;

  /* Output superclasses for message types */
  for ( m=thismodule->messages; m!=NULL; m=m->next ) {
    if (m->isExtern())
      continue;
    spew(top, CImessage, m->name);
  }

  /* Output Async Creation Methods for chares */
  for(c=thismodule->chares; c!=NULL; c=c->next) {
    // Do not emit creation method for main chare
    if(strcmp(c->name, "main")==0)
      continue;
    for(e=c->entries;e!=NULL;e=e->next) {
      if(strcmp(c->name, e->name)==0) {
        if(c->chareboc == CHARE) {
          spew(top, CIAsyncChareCreateProto, c->name, e->msgtype->name);
          spew(bot, CIAsyncChareCreateImpl, c->name, e->msgtype->name);
        } else {
          spew(top, CIAsyncGroupCreateProto, c->name, e->msgtype->name);
          spew(bot, CIAsyncGroupCreateImpl, c->name, e->msgtype->name);
        }
      }
    }
  }
  /* Output Proxy Classes for chares */
  for(c=thismodule->chares; c!=NULL; c=c->next) {
    if(c->chareboc == CHARE) {
      if(strcmp(c->name,"main")==0) {
        spew(top, CIProxyMainStart);
        for(e=c->entries;e!=NULL;e=e->next) {
          if(strcmp(c->name, e->name)==0) {
            continue;
          } else {
            // is a method, not constructor
            if(e->isReturnMsg()) {
              // method is a blocking method
              spew(top, CIProxyMainRetMethod, e->name, e->msgtype->name,
                        e->returnMsg->name);
            } else {
              // method is an ordinary method
              spew(top, CIProxyMainMethod, e->name, e->msgtype->name);
            }
          }
        }
        spew(top, CIProxyMainEnd);
        spew(bot, CIProxyMainDef);
      } else {
        spew(top, CIProxyClassStart, c->name);
        for(e=c->entries;e!=NULL;e=e->next) {
          if(strcmp(c->name, e->name)==0) {
            // Constructor
            spew(top, CIProxyClassConstructor, c->name, e->msgtype->name);
          } else {
            // is a method, not constructor
            if(e->isReturnMsg()) {
              // method is a blocking method
              spew(top, CIProxyClassRetMethod, c->name, e->name, 
                        e->msgtype->name, e->returnMsg->name);
            } else {
              // method is an ordinary method
              spew(top, CIProxyClassMethod, c->name, e->name, e->msgtype->name);
            }
          }
        }
        spew(top, CIProxyClassEnd);
      }
    } else {
      spew(top, CIProxyGroupStart, c->name);
      for(e=c->entries;e!=NULL;e=e->next) {
        if(strcmp(c->name, e->name)==0) {
          // Constructor
          spew(top, CIProxyGroupConstructor, c->name, e->msgtype->name);
        } else {
          // is a method, not constructor
          if(e->isReturnMsg()) {
            // method is a blocking method
            spew(top, CIProxyGroupRetMethod, c->name, e->name, e->msgtype->name,
                      e->returnMsg->name);
          } else {
            // method is an ordinary method
            spew(top, CIProxyGroupMethod, c->name, e->name, e->msgtype->name);
          }
        }
      }
      spew(top, CIProxyGroupEnd, c->name);
    }
  }
}

