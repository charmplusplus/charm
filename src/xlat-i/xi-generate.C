
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

  sprintf(definename, "CI_%s_TOP_H", thismodule->name);
  top << "#ifndef " << definename << "\n#define " << definename << endl;
  sprintf(definename, "CI_%s_BOT_H", thismodule->name);
  bot << "#ifndef " << definename << "\n#define " << definename << endl;

  GenerateStructsFns(top, bot) ;
  GenerateRegisterCalls(top, bot) ;

  top << "#endif\n";
  bot << "#endif\n";
}


void commonStuff(ofstream& top, ofstream& bot, Chare *c, Entry *e)
{
  char str[2048] ;

  /* This is the constructor EP */
  if ( strcmp(c->name, e->name) == 0 ) {

    sprintf(str,"\tnew (obj) %s((%s *)m) ;",
        c->name, e->msgtype->name) ;
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

  /* Output all chare and EP id variables. Note : if this chare is not
     defined in this module, put "extern" and dont initialize.  */

  for (c=thismodule->chares; c!=NULL; c=c->next ) {

    sprintf(str,"extern int _CK_chare_%s ;",c->name) ;
    top << str << endl ;
    if (!c->isExtern()){
      sprintf(str,"int _CK_chare_%s = 0 ;",c->name) ;
      bot << str << endl ;
    }

    for (e=c->entries; e!=NULL; e=e->next ) {
      sprintf(str,"extern int _CK_ep_%s_%s;",c->name,e->name) ;
      top << str << endl ;
      if (!c->isExtern()) {
        sprintf(str,"int _CK_ep_%s_%s = 0 ;",c->name,e->name) ;
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
        sprintf(str,"void _CK_call_threaded_%s_%s(void *velt)",
          c->name,e->name);
        bot << str << endl ;
        bot << "{" << endl ;
        bot << "\tElement *elt = (Element *) velt;" << endl;
        bot << "\tvoid *obj = elt->obj;" << endl;
        bot << "\tvoid *m = elt->m;" << endl;
        bot << "\tCpvAccess(currentChareBlock) = elt->chareblock;" << endl;

        commonStuff(top, bot, c, e);

        bot << "\tCmiFree(elt);" << endl;
        bot << "}" << endl ;

        sprintf(str,"void _CK_call_%s_%s(void *m, void *obj)",
          c->name,e->name);
        bot << str << endl ;
        bot << "{" << endl ;
        bot << "\tCthThread t;" << endl;
        bot << "\tElement *element = (Element *) CmiAlloc(sizeof(Element));" ;
	bot << endl;
        bot << "\telement->m = m;\n\telement->obj = obj;" << endl;
        bot << "\telement->chareblock = CpvAccess(currentChareBlock) ;"
            << endl;

        sprintf(str,
          "\tt = CthCreate( (void (*)(...)) _CK_call_threaded_%s_%s, (void *) element,%d);",
          c->name,e->name,e->get_stackSize()) ;
        bot << str << endl;
        bot << "\tCthSetStrategyDefault(t);" << endl;
        bot << "\tCthAwaken(t);" << endl;

        bot << "}" << endl ;

      } else { // NOT threaded
        sprintf(str,"void _CK_call_%s_%s(void *m, void *obj)",
          c->name,e->name);
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
    sprintf(str,
      "\t%s * m = (%s *)GenericCkAlloc(_CK_msg_%s,sizeof(%s),0) ;\n",
      m->name, m->name, m->name, m->name) ;
    bot << str ;
    bot << "\t(*out) = (void *) m ;" << endl ;
    bot << "\tm->unpack(in) ;\n}" << endl ;
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
  sprintf(str,"extern \"C\" void _CK_%s_init() ;\n",thismodule->name) ;
  bot << str ;
  sprintf(str,"void _CK_%s_init()\n{\n",thismodule->name) ;
  bot << str ;

/* first register all messages */
  for ( Message *m=thismodule->messages; m!=NULL; m=m->next ) {
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

    sprintf(str,"sizeof(%s)) ;\n\n",m->name) ;
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

      if ( chare->chareboc == CHARE ) 
        sprintf(str,
         "_CK_ep_%s_%s = registerEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, 1,",
         chare->name, ep->name, ep->name, chare->name,ep->name) ;
      else
       sprintf(str,
       "_CK_ep_%s_%s = registerBocEp(\"%s\", (FUNCTION_PTR)&_CK_call_%s_%s, 1,",
         chare->name, ep->name, ep->name, chare->name,ep->name) ;

      bot << str ;

      if ( strcmp(chare->name,"main")==0 && strcmp(ep->name,"main")==0 ) 
        sprintf(str,"0, _CK_chare_%s) ;\n\n",chare->name) ;
      else
        sprintf(str,"_CK_msg_%s, _CK_chare_%s) ;\n\n",
          ep->msgtype->name, chare->name) ;
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



