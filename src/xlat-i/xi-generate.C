
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
void GenerateRegisterCalls(ofstream& bot) ;
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
  GenerateRegisterCalls(bot) ;
  GenerateProxies(top, bot);

  spew(top, CIEndif);
  spew(bot, CIEndif);
}

static const char *CIBotConsEP = // charename, msgname
"  new (obj) \1((\2 *)m);\n"
;

static const char *CIBotRetEP = // charname, epname, msgname, retmsgname
"  ENVELOPE *env = ENVELOPE_UPTR(m);\n"
"  int i = GetEnv_ref(env);\n"
"  int j = GetEnv_pe(env);\n"
"  \4 *m2 = ((\1 *)obj)->\2((\3 *)m);\n"
"  SetRefNumber((void*) m2, i);\n"
"  CSendToFuture((void*) m2, j);\n"
;

static const char *CIBotRegularEP = // charename, epname, msgname
"  ((\1 *)obj)->\2((\3 *)m);\n"
;


void commonStuff(ofstream& bot, Chare *c, Entry *e)
{
  if ( strcmp(c->name, e->name) == 0 ) { // constructor EP
    spew(bot, CIBotConsEP, c->name, e->msgtype->name);
  } else if (e->isReturnMsg()){ // Returns a message
    spew(bot, CIBotRetEP, c->name,e->name,e->msgtype->name,e->returnMsg->name);
  } else { // Regular EP
    spew(bot, CIBotRegularEP, c->name, e->name, e->msgtype->name);
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

static const char *CITopChareDeclStart = // charename
"  public:\n"
"    static int _CK_charenum_\1;\n"
;

static const char *CITopChareDeclEnd = // charename
"};\n"
;

static const char *CIBotChareDef = // charename, modulename
"int _CK_chare_\1::_CK_charenum_\1 =  _CK_\2_id[0];\n"
;

static const char *CITopEPMDecl = // epname, msgname
"    static int \1_\2;\n"
;

static const char *CITopEPDecl = // epname
"    static int \1;\n"
;

static const char *CIBotEPMDef = // charename, epname, msgname
"int _CK_chare_\1::\2_\3  = 0;\n"
;

static const char *CIBotEPDef = // charename, epname
"int _CK_chare_\1::\2 = 0;\n"
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

static const char *CIBotCallThreadedEP1 = // charename, epname, msgname
"void _CK_call_threaded_\1_\2_\3(void *velt)\n"
"{\n"
"  _CI_Element_struct *elt = (_CI_Element_struct *) velt;\n"
"  void *obj = elt->obj;\n"
"  void *m = elt->m;\n"
"  CpvAccess(currentChareBlock) = elt->chareblock;\n"
;

static const char *CIBotCallThreadedEP2 = // charename, epname, msgname, stacksz
"  CmiFree(elt);\n"
"  CthFree(CthSelf());\n"
"  CthSuspend();\n"
"}\n"
"void _CK_call_\1_\2_\3(void *m, void *obj)\n"
"{\n"
"  CthThread t;\n"
"  _CI_Element_struct *element = \n"
"      (_CI_Element_struct *) CmiAlloc(sizeof(_CI_Element_struct));\n"
"  element->m = m;\n"
"  element->obj = obj;\n"
"  element->chareblock = CpvAccess(currentChareBlock);\n"
"  t = CthCreate( (CthVoidFn)_CK_call_threaded_\1_\2_\3,(void *)element,\4);\n"
"  CthSetStrategyDefault(t);\n"
"  CthAwaken(t);\n"
"}\n"
;

static const char *CIBotCallEP1 = // charename, epname, msgname
"void _CK_call_\1_\2_\3(void *m, void *obj)\n"
"{\n"
;

static const char *CIBotCallEP2 =
"}\n"
;

void GenerateStructsFns(ofstream& top, ofstream& bot)
{
  Chare *c; 
  Entry *e;

  spew(bot, CIBotStart);
  spew(top, CITopStart, thismodule->name);

  for (c=thismodule->chares; c!=0; c=c->next ) {
    if(c->isExtern())
      continue;
    top << "class _CK_chare_" << c->name << " ";
    if(c->numbases > 0) {
      top << ": ";
      int i;
      for(i=c->numbases-1;i>=0; i--) {
        top << "public _CK_chare_" << c->bases[i];
        if(i == 0)
          top << " ";
        else
          top << ", ";
      }
    }
    top << " {\n";
    spew(top, CITopChareDeclStart, c->name);
    spew(bot, CIBotChareDef, c->name, thismodule->name);
    for (e=c->entries; e!=0; e=e->next ) {
      if(e->isMessage()) {
        spew(top, CITopEPMDecl, e->name, e->msgtype->name);
        spew(bot, CIBotEPMDef, c->name, e->name, e->msgtype->name);
      } else {
        spew(top, CITopEPDecl, e->name);
        spew(bot, CIBotEPDef, c->name, e->name);
      }
    }
    spew(top, CITopChareDeclEnd, c->name);
  }

  /* Output EP stub functions. Note : we assume main::main always
     has argc-argv. */
  for ( c=thismodule->chares; c!=0; c=c->next ) {
    if(c->isExtern())
      continue;
    for (e=c->entries; e!=0; e=e->next ) {
      // If this is the main::main EP
      if (strcmp(c->name,"main")==0 && strcmp(e->name,"main")==0 ) {
        spew(bot, CIBotMainMainStub);
        moduleHasMain = 1;
        continue ;
      }
      // If this is a threaded EP
      if (e->isThreaded()){
        spew(bot, CIBotCallThreadedEP1, c->name, e->name, e->msgtype->name);
        commonStuff(bot, c, e);
        char str[16];
        sprintf(str, "%d", e->get_stackSize());
        spew(bot, CIBotCallThreadedEP2, c->name, e->name, e->msgtype->name,str);
      } else { // NOT threaded
        spew(bot, CIBotCallEP1, c->name, e->name, e->msgtype->name);
        commonStuff(bot, c, e);
        spew(bot, CIBotCallEP2);
      }
    }
  }


  ReadOnly *r;
  /* Output ids for readonly messages */
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if(r->isExtern())
      continue;
    if ( r->ismsg )
      spew(bot, CIBotROMIndex, r->name);
  }

  Message *m;
  /* Output ids for message types */
  for ( m=thismodule->messages; m!=0; m=m->next ) {
    spew(top, CITopMsgDecl, m->name);
    if (!m->isExtern())
      spew(bot, CIBotMsgDef, m->name);
  }

  /* for allocked MsgTypes output the alloc stub functions */
  for (m=thismodule->messages; m!=0; m=m->next) {
    if(m->isExtern())
      continue;
    spew(bot, CIBotMsgCoerceFn, m->name);
    if ( !m->allocked )
      continue ;
    spew(bot, CIBotMsgAllocFn, m->name);
  }

  /* for packable MsgTypes output the pack - unpack stub functions */
  for ( m=thismodule->messages; m!=0; m=m->next ) {
    if(m->isExtern())
      continue;
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
    if(r->isExtern())
      continue;
    if ( r->ismsg )
      spew(bot, CIBotFromROMCopy, r->name);
    else
      spew(bot, CIBotFromROCopy, r->name);
  }
  spew(bot, CIBotFromROEnd);

  spew(bot, CIBotToROStart, thismodule->name);
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if(r->isExtern())
      continue;
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
"  registerMainChare(_CK_chare_main::_CK_charenum_main,\n"
"                    _CK_chare_main::main, 1);\n"
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

static const char *CIBotRegisterPackedMsg = // msgname
"  _CK_msg_\1 = registerMsg(\"\1\", (FUNCTION_PTR)&GenericCkAlloc,\n"
"    (FUNCTION_PTR)&_CK_pack_\1, (FUNCTION_PTR)&_CK_unpack_\1,\n"
"    (FUNCTION_PTR) &_CK_coerce_\1, sizeof(\1));\n"
;

static const char *CIBotRegisterVarsizeMsg = //msgname
"  _CK_msg_\1 = registerMsg(\"\1\", (FUNCTION_PTR)&_CK_alloc_\1,\n"
"    (FUNCTION_PTR)&_CK_pack_\1, (FUNCTION_PTR)&_CK_unpack_\1,\n"
"    (FUNCTION_PTR) &_CK_coerce_\1, sizeof(\1));\n"
;

static const char *CIBotRegisterMsg = // msgname
"  _CK_msg_\1 = registerMsg(\"\1\", (FUNCTION_PTR)&GenericCkAlloc,\n"
"0, 0, (FUNCTION_PTR) &_CK_coerce_\1, sizeof(\1));\n"
;


static const char *CIBotRegisterChare = // charename
"  _CK_chare_\1::_CK_charenum_\1 = registerChare(\"\1\", sizeof(\1), 0);\n"
;

static const char *CIBotRegisterMainMain =
"  _CK_chare_main::main=registerEp(\"main\",\n"
"    (FUNCTION_PTR)&_CK_call_main_main,1,\n"
"    0, _CK_chare_main::_CK_charenum_main);\n"
;

static const char *CIBotRegisterChareEP = //charename, epname, msgname
"  _CK_chare_\1::\2_\3 = registerEp(\"\2\", (FUNCTION_PTR)&_CK_call_\1_\2_\3,\n"
"    1, _CK_msg_\3, _CK_chare_\1::_CK_charenum_\1);\n"
;

static const char *CIBotRegisterBocEP = //bocname, epname, msgname
"  _CK_chare_\1::\2_\3=registerBocEp(\"\2\",(FUNCTION_PTR)&_CK_call_\1_\2_\3,\n"
"    1, _CK_msg_\3, _CK_chare_\1::_CK_charenum_\1);\n"
;

/* now register readonlies and readonli messages */
void GenerateRegisterCalls(ofstream& bot)
{
  spew(bot, CIBotRegisterStart, thismodule->name);

/* first register all messages */
  for ( Message *m=thismodule->messages; m!=0; m=m->next ) {
    if(m->isExtern())
      continue;
    if( m->allocked) {
      spew(bot, CIBotRegisterVarsizeMsg, m->name);
      continue;
    } else if (!m->packable)  {
      spew(bot, CIBotRegisterMsg, m->name);
      continue;
    } else {
      spew(bot, CIBotRegisterPackedMsg, m->name);
    }
  }

/* now register all chares and BOCs and their EPs */
  for ( Chare *chare=thismodule->chares; chare!=0; chare=chare->next ) {
    if(chare->isExtern())
      continue;
    spew(bot, CIBotRegisterChare, chare->name);
    for  ( Entry *ep=chare->entries; ep!=0; ep=ep->next ) {
      if(strcmp(chare->name, "main")==0 && strcmp(ep->name, "main")==0) {
        spew(bot, CIBotRegisterMainMain);
        continue;
      } else if(chare->chareboc == CHARE) {
        spew(bot, CIBotRegisterChareEP, chare->name, ep->name, 
             ep->msgtype->name);
        continue;
      } else {
        spew(bot, CIBotRegisterBocEP, chare->name, ep->name, ep->msgtype->name);
        continue;
      }
    }
  }

  if (moduleHasMain)
    spew(bot, CIBotRegisterMainChare);


/* register distributed-table-variables */
  for ( Table *t=thismodule->tables; t!=0; t=t->next ) {
    if(t->isExtern())
      continue;
    spew(bot, CIBotRegisterTable, t->name);
  }

  spew(bot, CIBotRegisterROStart);
  ReadOnly *r;
  for ( r=thismodule->readonlys; r!=0; r=r->next ) {
    if(r->isExtern())
      continue;
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
"  protected:\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(ChareIDType _cid) { _ck_cid = _cid; };\n"
;

static const char *CIProxyClassConstructor = // classname, msgname
"    CProxy_\01(\02 *m, int pe=CK_PE_ANY) {\n"
"      new_chare2(\01, \02, m, &_ck_cid, pe);\n"
"    };\n"
"    int _ptr_\01(\02 *m) {\n"
"      return GetEntryPtr(\01,\01,\02);\n"
"    };\n"
;

static const char *CIProxyClassMethod = // classname, methodname, msgname
"    void \02(\03 *m) {\n"
"      CSendMsg(\01, \02, \03, m, &_ck_cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyClassRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m) {\n"
"      return (\04 *) CRemoteCallFn(GetEntryPtr(\01,\02,\03), m, &_ck_cid);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyClassEnd =
"    ChareIDType _getCid(void) { return _ck_cid; }; \n"
"    void _setCid(ChareIDType _cid) { _ck_cid = _cid; }; \n"
"};\n"
;

static const char *CIProxyMainStart =
"class main;\n"
"class CProxy_main {\n"
"  private:\n"
"    int dummy;\n"
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
"  protected:\n"
"    CProxy_\01() {};\n"
"  public:\n"
"    CProxy_\01(int _bocid) { _ck_bocid = _bocid; };\n"
;

static const char *CIProxyGroupConstructor = // classname, msgname
"    CProxy_\01(\02 *m) {\n"
"      _ck_bocid = new_group(\01, \02, m);\n"
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
"      CBroadcastMsgBranch(\01, \02, \03, m, _ck_bocid);\n"
"    };\n"
"    void \02(\03 *m, int onPE) {\n"
"      CSendMsgBranch(\01, \02, \03, m, _ck_bocid, onPE);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyGroupRetMethod = // classname, methodname, msgname, retmsg
"    \04 *\02(\03 *m, int onPE) {\n"
"      return (\04 *) CRemoteCallBranchFn(GetEntryPtr(\01,\02,\03), m, _ck_bocid, onPE);\n"
"    };\n"
"    int _ptr_\02(\03 *m) {\n"
"      return GetEntryPtr(\01,\02,\03);\n"
"    };\n"
;

static const char *CIProxyGroupEnd = // classname
"    int _getBocid(void) { return _ck_bocid; }; \n"
"    void _setBocid(int _bocid) { _ck_bocid = _bocid; }; \n"
"    \01 *_localBranch(void) {\n"
"      return (\01 *) CLocalBranch(\01, _ck_bocid);\n"
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
    if(c->isExtern())
      continue;
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
    if(c->isExtern())
      continue;
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
        top << "class " << c->name << ";\n";
        top << "class CProxy_" << c->name << " : public virtual _CK_CID";
        if(c->numbases > 0) {
          int i;
          for(i=c->numbases-1;i>=0; i--) {
            top << ", public CProxy_" << c->bases[i];
          }
        }
        top << " {\n";
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
      top << "class " << c->name << ";\n";
      top << "class CProxy_" << c->name << " : public virtual _CK_GID";
      if(c->numbases > 0) {
        int i;
        for(i=0;i<c->numbases; i++) {
          top << ", public CProxy_" << c->bases[i];
        }
      }
      top << " {\n";
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

