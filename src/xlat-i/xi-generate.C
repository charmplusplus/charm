
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


static void
spew(ofstream &strm, const char *ostr,
     char *a1 = "ERROR", char *a2 = "ERROR",
     char *a3 = "ERROR", char *a4 = "ERROR",
     char *a5 = "ERROR")
{
  int i;
  for(i=0; i<strlen(ostr); i++){
    switch(ostr[i]){
      case '\01': strm << a1; break;
      case '\02': strm << a2; break;
      case '\03': strm << a3; break;
      case '\04': strm << a4; break;
      case '\05': strm << a5; break;
      default: strm << ostr[i];
    }
  }
}

static const char *CITopIfndef = // moduleName
"#ifndef CI_\1_TOP_H\n"
"#define CI_\1_TOP_H\n"
"\n"
;

static const char *CIBotIfndef = // moduleName
"#ifndef CI_\1_BOT_H\n"
"#define CI_\1_BOT_H\n"
"\n"
;

static const char *CIEndif = 
"#endif\n"
;

void Generate(char *interfacefile)
{
  char modulename[1024], topname[1024], botname[1024];
  strcpy(modulename, interfacefile) ;
  modulename[strlen(interfacefile)-3] = '\0' ; // assume ModuleName.ci
  sprintf(topname,"%s.top.h", modulename) ;
  sprintf(botname,"%s.bot.h", modulename) ;

  ofstream top(topname), bot(botname) ;

  if (top == 0 || bot == 0) {
    cerr << "Cannot open " << topname 
         << " or " << botname << " for writing !!" << endl;
    exit(1);
  }

  spew(top, CITopIfndef, thismodule->name);
  spew(bot, CIBotIfndef, thismodule->name);

  GenerateStructsFns(top, bot) ;
  GenerateRegisterCalls(top, bot) ;
  GenerateProxies(top, bot);

  spew(top, CIEndif);
  spew(bot, CIEndif);
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

static const char *CIBotStart = 
"typedef struct {\n"
"  void *obj;\n"
"  void *m;\n"
"  CHARE_BLOCK *chareblock;\n"
"} _CI_Element_struct;\n"
"CpvExtern(CHARE_BLOCK *,currentChareBlock);\n"
;

static const char *CITopStart = // modulename
"extern char *_CK_\1_id;\n"
;

static const char *CITopChareDecl = // charename
"extern int _CK_chare_\1;\n"
;

static const char *CIBotChareDef = // charename, modulename
"int _CK_chare_\1 =  _CK_\2_id[0];\n"
;

static const char *CITopEPMDecl = // charename, epname, msgname
"extern int _CK_ep_\1_\2_\3;\n"
;

static const char *CITopEPDecl = // charename, epname
"extern int _CK_ep_\1_\2;\n"
;

static const char *CIBotEPMDef = // charename, epname, msgname
"int _CK_ep_\1_\2_\3  = 0;\n"
;

static const char *CIBotEPDef = // charename, epname
"int _CK_ep_\1_\2 = 0;\n"
;

static const char *CIBotMainMainStub = 
"extern \"C\"\n"
"void _CK_call_main_main(void *m,void *obj,int argc,char *argv[]);\n"
"void _CK_call_main_main(void *m,void *obj,int argc,char *argv[])\n"
"{\n"
"  new (obj) main(argc,argv);\n"
"}\n"
"\n"
;

static const char *CIBotFromROStart = // moduleName
"extern \"C\" void _CK_13CopyFromBuffer(void *,int);\n"
"void _CK_\1_CopyFromBuffer(void **_CK_ReadMsgTable)\n"
"{\n"
;

static const char *CIBotFromROCopy = //readonlyName
"  _CK_13CopyFromBuffer(&\1,sizeof(\1)) ;\n"
;

static const char *CIBotFromROMCopy = //readonlyMessageName
"  {\n"
"    void **temp = (void **)(&\1);\n"
"    *temp = _CK_ReadMsgTable[_CK_index_\1];\n"
"  }\n"
;

static const char *CIBotFromROEnd =
"}\n"
;

static const char *CIBotROMIndex = // readonlyMessageName
"int _CK_index_\1;\n"
;

static const char *CITopMsgDecl = // messageName
"extern int _CK_msg_\1;\n"
;

static const char *CIBotMsgDef = // messageName
"int _CK_msg_\1 = 0;\n"
;

static const char *CIBotMsgCoerceFn = // msgName
"static comm_object *_CK_coerce_\1(void *msg)\n"
"{\n"
"  return (comm_object *) new (msg) \1;\n"
"}\n"
;

static const char *CIBotMsgAllocFn = // msgName
"static void *_CK_alloc_\1(int msgno, int size, int *array, int prio)\n"
"{\n"
"  void *out;\n"
"  out = \1::alloc(msgno,size,array,prio);\n"
"  return out;\n"
"}\n"
;

static const char *CIBotMsgPackFn = // msgName
"static void _CK_pack_\1(void *in, void **out, int *length)\n"
"{\n"
"  (*out) = ((\1 *)in)->pack(length);\n"
"}\n"
;

static const char *CIBotMsgUnpackFn = // msgName
"static void _CK_unpack_\1(void *in, void **out)\n"
"{\n"
"  (*out) = (void *) in;\n"
"  ((\1 *)(*out))->unpack(in);\n}"
"\n"
;

static const char *CIBotMsgUnpackFnWithAlloc = // msgName
"static void _CK_unpack_\1(void *in, void **out)\n"
"{\n"
"  \1 * m = (\1 *)GenericCkAlloc(_CK_msg_\1,sizeof(\1),0);\n"
"  (*out) = (void *) m;\n"
"  m->unpack(in);\n"
"}\n"
;

static const char *CIBotToROStart = // moduleName
"extern \"C\" void _CK_13CopyToBuffer(void *,int);\n"
"extern \"C\" void ReadMsgInit(void *,int);\n"
"void _CK_\1_CopyToBuffer(void)\n"
"{\n"
;

static const char *CIBotToROCopy = // readonlyName
"  _CK_13CopyToBuffer(&\1,sizeof(\1));\n"
;

static const char *CIBotToROMCopy = // readonlyMsgName
"  ReadMsgInit(\1,_CK_index_\1);\n"
;

static const char *CIBotToROEnd =
"}\n"
;
void GenerateStructsFns(ofstream& top, ofstream& bot)
{
  char str[2048] ;
  Chare *c; 
  Entry *e;

  spew(bot, CIBotStart);
  spew(top, CITopStart, thismodule->name);

  for (c=thismodule->chares; c!=0; c=c->next ) {
    spew(top, CITopChareDecl, c->name);
    if (!c->isExtern())
      spew(bot, CIBotChareDef, c->name, thismodule->name);
    for (e=c->entries; e!=0; e=e->next ) {
      if(e->isMessage())
        spew(top, CITopEPMDecl, c->name, e->name, e->msgtype->name);
      else
        spew(top, CITopEPDecl, c->name, e->name);
      if (!c->isExtern())
        if(e->isMessage())
          spew(bot, CIBotEPMDef, c->name, e->name, e->msgtype->name);
        else
          spew(bot, CIBotEPDef, c->name, e->name);
    }
  }



  /* Output EP stub functions. Note : we assume main::main always
     has argc-argv. */
  for ( c=thismodule->chares; c!=0; c=c->next ) {
    for (e=c->entries; e!=0; e=e->next ) {
      // If this is the main::main EP
      if (strcmp(c->name,"main")==0 && strcmp(e->name,"main")==0 ) {
        spew(bot, CIBotMainMainStub);
        moduleHasMain = 1;
        continue ;
      }
      // If this is a threaded EP
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
        bot << "\t_CI_Element_struct *elt = (_CI_Element_struct *) velt;" << endl;
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
        bot << "\t_CI_Element_struct *element = (_CI_Element_struct *) CmiAlloc(sizeof(_CI_Element_struct));" ;
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
  for ( r=thismodule->readonlys; r!=0; r=r->next ) 
    if ( r->ismsg )
      spew(bot, CIBotROMIndex, r->name);

  Message *m;
  /* Output ids for message types */
  for ( m=thismodule->messages; m!=0; m=m->next ) {
    spew(top, CITopMsgDecl, m->name);
    if (!m->isExtern())
      spew(bot, CIBotMsgDef, m->name);
  }

  /* for allocked MsgTypes output the alloc stub functions */
  for (m=thismodule->messages; m!=0; m=m->next) {
    spew(bot, CIBotMsgCoerceFn, m->name);
    if ( !m->allocked )
      continue ;
    spew(bot, CIBotMsgAllocFn, m->name);
  }

  /* for packable MsgTypes output the pack - unpack stub functions */
  for ( m=thismodule->messages; m!=0; m=m->next ) {
    if ( !m->packable )
      continue ;
    spew(bot, CIBotMsgPackFn, m->name);
    if (!m->allocked)  // Varsize message don't need alloc
      spew(bot, CIBotMsgUnpackFnWithAlloc, m->name);
    else
      spew(bot, CIBotMsgUnpackFn, m->name);
  }


  spew(bot, CIBotFromROStart, thismodule->name);
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if ( r->ismsg )
      spew(bot, CIBotFromROMCopy, r->name);
    else
      spew(bot, CIBotFromROCopy, r->name);
  }
  spew(bot, CIBotFromROEnd);

  spew(bot, CIBotToROStart, thismodule->name);
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if ( r->ismsg )
      spew(bot, CIBotToROMCopy, r->name);
    else
      spew(bot, CIBotToROCopy, r->name);
  }
  spew(bot, CIBotToROEnd);
}


static const char *CIBotRegisterStart = // moduleName
"char *_CK_\1_id=\"\\0charmc autoinit \1\";\n"
"extern \"C\" void _CK_\1_init(void);\n"
"void _CK_\1_init(void)\n"
"{\n"
;

static const char *CIBotRegisterMainChare =
"  registerMainChare(_CK_chare_main, _CK_ep_main_main, 1);\n"
;

static const char *CIBotRegisterTable = // tableName
"  \1.SetId(registerTable(\"\1\", 0, 0));\n"
;

static const char *CIBotRegisterEnd =
"}\n"
;

static const char *CIBotRegisterROStart = 
"  int readonlysize = 0;\n"
;

static const char *CIBotRegisterROCalculate = // readonlyName
"  readonlysize += sizeof(\1);\n"
;

static const char *CIBotRegisterROM = // readonlyMsgName
"  _CK_index_\1 = registerReadOnlyMsg();\n"
;

static const char *CIBotRegisterRO = // moduleName
"  registerReadOnly(readonlysize, (FUNCTION_PTR)&_CK_\1_CopyFromBuffer,\n"
"                                 (FUNCTION_PTR)&_CK_\1_CopyToBuffer);\n"
;

/* now register readonlies and readonli messages */
void GenerateRegisterCalls(ofstream& top, ofstream& bot)
{
  char str[2048] ;

  spew(bot, CIBotRegisterStart, thismodule->name);

/* first register all messages */
  for ( Message *m=thismodule->messages; m!=0; m=m->next ) {
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
  for ( Chare *chare=thismodule->chares; chare!=0; chare=chare->next )
  {
    sprintf(str,"_CK_chare_%s = registerChare(\"%s\", sizeof(%s), 0) ;\n\n",
      chare->name,chare->name,chare->name) ;
    bot << str ;

    for  ( Entry *ep=chare->entries; ep!=0; ep=ep->next ) {

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
    } // end for ep =
  } // end for chare =
  bot << "\n\n" ;

  if (moduleHasMain)
    spew(bot, CIBotRegisterMainChare);


/* register distributed-table-variables */
  for ( Table *t=thismodule->tables; t!=0; t=t->next )
    spew(bot, CIBotRegisterTable, t->name);

  spew(bot, CIBotRegisterROStart);
  ReadOnly *r;
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if ( r->ismsg )
      spew(bot, CIBotRegisterROM, r->name);
    else
      spew(bot, CIBotRegisterROCalculate, r->name);
  }
  spew(bot, CIBotRegisterRO, thismodule->name);
  spew(bot, CIBotRegisterEnd);
}

static const char *CImessage = // msgType
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
"    void *_packbuffer(int len) {\n"
"      return new_packbuffer((void *) this, len);\n"
"    }\n"
"};\n"
;

static const char *CIAsyncChareCreateProto = // charename, msgname
"extern void CNew_\01(\02 *m, int onPE=CK_PE_ANY);\n"
;

static const char *CIAsyncChareCreateImpl = // charename, msgname
"void CNew_\01(\02 *m, int onPE) {\n"
"  new_chare2(\01, \02, m, (ChareIDType *)0, onPE);\n"
"};\n"
;

static const char *CIAsyncGroupCreateProto = // groupname, msgname
"extern int CNew_\01(\02 *m);\n"
;

static const char *CIAsyncGroupCreateImpl = // groupname, msgname
"int CNew_\01(\02 *m) {\n"
"  return new_group(\01, \02, m);\n"
"};\n"
;

static const char *CIProxyClassStart = // classname
"class \01;\n"
"class CProxy_\01 {\n"
"  private:\n"
"    ChareIDType cid;\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(ChareIDType _cid) { cid = _cid; };\n"
;

static const char *CIProxyClassConstructor = // classname, msgname
"    CProxy_\01(\02 *m, int pe=CK_PE_ANY) {\n"
"      new_chare2(\01, \02, m, &cid, pe);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(\01,\01,\02);\n"
"    };\n"
;

static const char *CIProxyClassMethod = // classname, methodname, msgname
"    void \02(\03 *m) {\n"
"      CSendMsg(\01, \02, \03, m, &cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyClassRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m) {\n"
"      return (\04 *) CRemoteCallFn(GetEntryPtr(\01,\02,\03), m, &cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyClassEnd =
"    ChareIDType _getCid(void) { return cid; }; \n"
"    void _setCid(ChareIDType _cid) { cid = _cid; }; \n"
"};\n"
;

static const char *CIProxyMainStart =
"class main;\n"
"class CProxy_main {\n"
"  private:\n"
"  int dummy;\n"
"  public:\n"
"    CProxy_main() {};\n"
;

static const char *CIProxyMainMethod = // methodname, msgname
"    void \01(\02 *m) {\n"
"      CSendMsg(main, \01, \02, m, &mainhandle);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(main,\01,\02);\n"
"    };\n"
;

static const char *CIProxyMainRetMethod = // methodname, msgname, retmsg
"    \03 *\01(\02 *m) {\n"
"      return (\03 *) CRemoteCallFn(GetEntryPtr(main,\01,\02), m, &mainhandle);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(main,\01,\02);\n"
"    };\n"
;

static const char *CIProxyMainEnd =
"};\n"
"extern CProxy_main mainproxy;\n"
;

static const char *CIProxyMainDef =
"CProxy_main mainproxy;\n"
;

static const char *CIProxyGroupStart = // classname
"class \01;\n"
"class CProxy_\01 {\n"
"  private:\n"
"    int bocid;\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(int _bocid) { bocid = _bocid; };\n"
;

static const char *CIProxyGroupConstructor = // classname, msgname
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

static const char *CIProxyGroupMethod = // classname, methodname, msgname
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

static const char *CIProxyGroupRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m, int onPE) {\n"
"      return (\04 *) CRemoteCallBranchFn(GetEntryPtr(\01,\02,\03), m, bocid, onPE);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyGroupEnd = // classname
"    int _getBocid(void) { return bocid; }; \n"
"    void _setBocid(int _bocid) { bocid = _bocid; }; \n"
"    \01 *_localBranch(void) {\n"
"      return (\01 *) CLocalBranch(\01, bocid);\n"
"    };\n"
"};\n"
;

void GenerateProxies(ofstream& top, ofstream& bot)
{
  Message *m;
  Chare *c;
  Entry *e;

  /* Output superclasses for message types */
  for ( m=thismodule->messages; m!=0; m=m->next ) {
    if (m->isExtern())
      continue;
    spew(top, CImessage, m->name);
  }

  /* Output Async Creation Methods for chares */
  for(c=thismodule->chares; c!=0; c=c->next) {
    // Do not emit creation method for main chare
    if(strcmp(c->name, "main")==0)
      continue;
    for(e=c->entries;e!=0;e=e->next) {
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
  for(c=thismodule->chares; c!=0; c=c->next) {
    if(c->chareboc == CHARE) {
      if(strcmp(c->name,"main")==0) {
        spew(top, CIProxyMainStart);
        for(e=c->entries;e!=0;e=e->next) {
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
        for(e=c->entries;e!=0;e=e->next) {
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
      for(e=c->entries;e!=0;e=e->next) {
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

