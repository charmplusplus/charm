#include <stdio.h>
extern int InPass1;
extern char *CkLocalPtr;
extern char *DataSuffix;
extern char *AssignMyDataPtr;
extern char *CkMyData;
char *Map();
int  writeoutput(),WriteReturn();

static reverse_elist();
static reverse_clist();
static init_code();
static entry_header_code();
static declare_lptr();
static assign_lptr();
static emit_expect_code();
static fill_charenum();
/* static emit_epconv(); */

#define PRINT_BUFFER 256

/* defined in dagger.h and dag.h also */
#define _dag3_WLIMIT 8 
/* define in dagger.h also */ 
#define DAG_NOREF   0

#ifndef NULL
#define NULL 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define NOFREE 0
#define EXIT 1

#define MATCH    1
#define MULTIPLE 2
#define ANY      4
#define AUTOFREE 8

#define IS_MULTIPLE(eptr)      (eptr->etype & MULTIPLE)
#define IS_MATCHING(eptr)      (eptr->etype & MATCH)
#define IS_MATCHING_WHEN(wptr) (wptr->wtype & MATCH)
#define IS_ANY(wptr)           (wptr->wtype & ANY)

struct S_DLIST {  /* list of dag chares */
    struct S_ELIST *e_list;
    struct S_WLIST *w_list;
    struct S_CLIST *c_list;
    int   numofwhen;
    int   numofentry;
    int   is_a_chare;
    char  *charename;
    struct S_DLIST *next;
};

struct S_ELIST {  /* list of entry points */
    char  *name;
    char  *msgname;
    char  *varname;
    char  *msgtype;
    int   eno;
    int   etype;            /* MATCH/NONMATCH,MULTIPLE/ORDINARY */
    int   numofwhen;
    struct S_EWLIST *wlist;
    struct S_ELIST *next;
};

/* list of when nodes that contains the enrypoint */
struct S_EWLIST {
    int             position; 
    struct S_WLIST  *wnode;
    struct S_EWLIST *next;
};

/* list of when blocks */
struct S_WLIST {
    int      wno;
    int      wtype;           /* ANY,MATCHING,OTHER */
    int      count;
    int      position_count;
    struct S_WCONDLIST *clist;
    struct S_WLIST    *next;
};


/* list of conditions in the when  block */
struct S_WCONDLIST {
    int      position;
    int      isentry;               /*  ENTRY/CONDITIONVAR */
    void     *eptr;
    struct S_WCONDLIST *next;
}; 

/* list of condition variables */
struct S_CLIST {
    char     *name;
    int      cno;
    struct S_EWLIST *wlist;
    struct S_CLIST *next;
};


typedef struct S_DLIST DLIST;
typedef struct S_ELIST ELIST;
typedef struct S_WLIST WLIST;
typedef struct S_CLIST CLIST;
typedef struct S_EWLIST EWLIST;
typedef struct S_WCONDLIST WCONDLIST;



static ELIST *is_entry();
static CLIST *insert_cond();


char *wcounter="_dag3_wcounter";
char *wcount="_dag3_wcount";
char *etype="_dag3_etype";
char *epbuffer="_dag3_ep_buffer";
char *rl="_dag3_rl";
char *current_refnum = "_dag3_myrefnum";
char *condvar_fptr = "_dag3_cv_fptr";
char *efunc_fptr   = "_dag3_epf_fptr";
char *wswitch_fptr = "_dag3_wswitch_fptr";
char *ischare      = "_dag3_ischare";
char *activator    = "_dag3_activator";
char *penum        = "_dag3_penum";
char *cid          = "_dag3_cid";
char *epconv       = "_dag3_epconv";
char *flist        = "_dag3_flist";
#ifdef STRACE
char *mw_fptr       = "_dag3_mw_fptr";
#endif
static char modulename[256];
static char *charenum,trace_charename[512];

DLIST *d_list_head,*d_list_tail;
ELIST *e_list;
WLIST *w_list_head, *w_list_tail;
CLIST *c_list; 
DLIST *current_dag;
WLIST *current_when;
int   current_wno,current_eno,current_cno;
char  temp[PRINT_BUFFER];
char  base_name[PRINT_BUFFER];
ELIST *savedeptr;
int   first_cond_flag;

_dag_init(mname)
char *mname;
{
      if (!InPass1) return;
      d_list_head = d_list_tail = NULL;
      strcpy(modulename,mname);
}


_dag_begindag(charename,is_a_chare)
char *charename;
int is_a_chare;
{  
      DLIST *dag;

      if (!InPass1) return;

      dag       = (DLIST *)malloc(sizeof(DLIST));
      dag->next = NULL;
      dag->charename = (char *)malloc(strlen(charename)+1);
      strcpy(dag->charename,charename);
      if (d_list_head == NULL) 
         d_list_head = d_list_tail = dag;
      else {
         d_list_tail->next = dag;
         d_list_tail = dag;
      }
      e_list      = (ELIST *) NULL;
      w_list_head = w_list_tail = (WLIST *) NULL;
      c_list      = (CLIST *) NULL;
      current_wno = current_eno = current_cno = 0;
      dag->numofwhen = dag->numofentry = 0;
      dag->is_a_chare = is_a_chare;
}

_dag_enddag()
{
      if (!InPass1) return;
      d_list_tail->e_list = e_list;
      d_list_tail->c_list = c_list;
      d_list_tail->w_list = w_list_head;
      d_list_tail->numofentry= current_eno;
      d_list_tail->numofwhen = current_wno;
}

_dag_newentry(name)
char *name;
{
     ELIST *new_entry;
     int   size;

     if (!InPass1) return;
     new_entry       = (ELIST *)malloc(sizeof(ELIST));
     new_entry->next = e_list;
     e_list          = new_entry;
     size            = strlen(name)+1;
     e_list->name    = (char *)malloc(strlen(name)+1);
     strcpy(e_list->name,name);
     e_list->wlist   = NULL;
     e_list->eno     = current_eno++;
     e_list->etype   = 0;
     e_list->wlist   = (EWLIST *) NULL;
     e_list->numofwhen = 0;
     if (e_list->eno == 0) sprintf(base_name,"%s",name);
}

_dag_entrytype(varname)
char *varname;
{
     int size;

     if (!InPass1) return;
     if (varname != NULL) {
         size = strlen(varname) + 1;
         e_list->varname = (char*) malloc(size);
         strcpy(e_list->varname,varname); 
         e_list->etype |= MULTIPLE;
        }
     else 
        e_list->varname = NULL; 
}

      

_dag_entrymsg(msgtype,msgname)
char *msgtype,*msgname;
{    int size;
     if (!InPass1) return;
     size            = strlen(msgname)+1;
     e_list->msgname = (char *)malloc(size);
     strcpy(e_list->msgname,msgname);
     size            = strlen(msgtype) + 1;
     e_list->msgtype = (char *)malloc(size);
     strcpy(e_list->msgtype,msgtype);
}


_dag_entry_match(flag)
int flag;
{ 
    if (!InPass1) return;
    if (flag) e_list->etype |= MATCH; 
}

_dag_entry_free(flag)
{
    if (!InPass1) return;
    if (flag) e_list->etype |= AUTOFREE;
}

_dag_newwhen()
{
     WLIST *new_when; 
     if (!InPass1) return;
     new_when = (WLIST *) malloc(sizeof(WLIST));
     if (w_list_head == NULL) 
        w_list_head = w_list_tail = new_when;
     else {
        w_list_tail->next =  new_when;
        w_list_tail = new_when;
     } 
     new_when->clist = NULL;
     new_when->position_count = 0;
     new_when->wno   = current_wno++;
     new_when->wtype = 0;
     new_when->count = 0;
     first_cond_flag = TRUE;
}    




_dag_when_cond(condname)
char *condname;
{
     WCONDLIST *cond;
     ELIST *eptr;
     CLIST *cptr;
     EWLIST *temp2;

     if (!InPass1) return;

     cond = (WCONDLIST *) malloc(sizeof(WCONDLIST));
     cond->next = w_list_tail->clist;
     w_list_tail->clist = cond;
     w_list_tail->count++;
     if (eptr = is_entry(condname)) {
        cond->isentry = TRUE;
        cond->eptr  = (void *) eptr;
        cond->position = w_list_tail->position_count++;
        temp2 = (EWLIST *) malloc(sizeof(EWLIST));
        temp2->next = eptr->wlist;
        temp2->wnode= w_list_tail;
        temp2->position = cond->position;
        eptr->wlist = temp2;
        if ( !IS_ANY(w_list_tail) )  eptr->numofwhen++;
        if ( IS_MATCHING(eptr) ) {
           if (first_cond_flag)
              w_list_tail->wtype |= MATCH;
           else if ( !IS_MATCHING_WHEN(w_list_tail) )
              printf("dag:error : mixed entry points\n"); 
          }
        if (first_cond_flag) first_cond_flag = FALSE;
        }
     else {
        cond->isentry = FALSE;
        cptr = insert_cond(condname);
        cond->eptr = (void *) cptr;
        temp2 = (EWLIST *) malloc(sizeof(EWLIST));
        temp2->next = cptr->wlist;
        temp2->wnode= w_list_tail;
        cptr->wlist = temp2;
     }
}


_dag_when_any(any)
int any;
{
     if (!InPass1) return;
     if (any) w_list_tail->wtype |= ANY;
} 



_dag2_begindag()
{
    /* set the current dag chare */
    if (InPass1) return;
    current_dag = d_list_head;
    current_when = NULL;
    if (current_dag == NULL) {
       error("dag: error\n",EXIT);
    }
    d_list_head = d_list_head->next;
    reverse_elist();
    reverse_clist();
    fill_charenum(current_dag->charename);
#ifdef STRACE
    write_depn();
#endif
}




_dag2_decl()
{

    if (InPass1) return;

    /* declare control data */
    sprintf(temp,"int %s[%d];",etype,current_dag->numofentry);
    writeoutput(temp,NOFREE);
    sprintf(temp,"_dag3_BUFFER *%s[%d];",epbuffer,current_dag->numofentry);
    writeoutput(temp,NOFREE); WriteReturn();
    sprintf(temp,"int %s[%d];",epconv,current_dag->numofentry);
    writeoutput(temp,NOFREE); WriteReturn();
    sprintf(temp,"int %s[%d];",wcount,current_dag->numofwhen);
    writeoutput(temp,NOFREE);
    sprintf(temp,"_dag3_COUNT *%s[%d];",wcounter,current_dag->numofwhen);
    writeoutput(temp,NOFREE); WriteReturn();
    sprintf(temp,"_dag3_RL %s;",rl);
    writeoutput(temp,NOFREE);  WriteReturn();
    sprintf(temp,"int %s;",current_refnum);
    writeoutput(temp,NOFREE);  WriteReturn();
    sprintf(temp,"int (*%s)(),(*%s)(), (*%s)();",
            condvar_fptr,efunc_fptr,wswitch_fptr);
    writeoutput(temp,NOFREE); WriteReturn(); 
#ifdef STRACE
    sprintf(temp,"int (*%s)();",mw_fptr);
    writeoutput(temp,NOFREE); WriteReturn();
#endif
    sprintf(temp,"int %s,%s; _dag3_FREELIST %s;",ischare,activator,flist);
    writeoutput(temp,NOFREE); 
    sprintf(temp,"ChareIDType %s;",cid);
    writeoutput(temp,NOFREE); WriteReturn();
    if (!current_dag->is_a_chare) {
       sprintf(temp,"int %s;",penum);
       writeoutput(temp,NOFREE); WriteReturn();
    }
}

_dag2_cond_code()
{
    CLIST *cond;
    EWLIST *when_set;

    if (InPass1) return;

    cond = current_dag->c_list;
    sprintf(temp,"static _dag7%s(%s,var,flag,refnum)",current_dag->charename,CkLocalPtr); 
    writeoutput(temp,NOFREE);WriteReturn();
    declare_lptr();
    writeoutput("_dag3_DAGVAR *var; int flag,refnum;{",NOFREE); WriteReturn();
    writeoutput("_dag3_COUNT *counter; int dispatched;",NOFREE); WriteReturn();
#ifdef STRACE
    writeoutput("unsigned int t2;",NOFREE); WriteReturn();
#endif
    assign_lptr();

#ifdef STRACE
    writeoutput("t2=dag_timer();",NOFREE); WriteReturn();
#endif

    writeoutput("if (flag == 0) --var->counter;",NOFREE); WriteReturn();
    writeoutput("if (flag == 1 || var->counter==0) {",NOFREE); WriteReturn();
    writeoutput("dispatched = 0;",NOFREE); WriteReturn();


    writeoutput("  switch (var->index) {",NOFREE); WriteReturn();
    while(cond) {  
      sprintf(temp,"     case %d : ",cond->cno);
      writeoutput(temp,NOFREE); WriteReturn();
      when_set = cond->wlist;
      while(when_set) {
      
        writeoutput ("if (flag == 0)",NOFREE);
        sprintf(temp,"counter = _dag4_nonmfc(&(%s%s),%s%s,%s%s,%d);",
                CkMyData,flist,
                CkMyData,wcount,CkMyData,wcounter,when_set->wnode->wno);
        writeoutput(temp,NOFREE); WriteReturn();

        sprintf(temp,"else counter = _dag4_mfc(&(%s%s),%s%s,%s%s,%d,refnum);",
                CkMyData,flist,
                CkMyData,wcount,CkMyData,wcounter,when_set->wnode->wno);
        writeoutput(temp,NOFREE); WriteReturn();

        writeoutput("if (--(counter->value) == 0){",NOFREE);
        /* refnum field was zero, if doesn't work , undo it */
        sprintf(temp," _dag4_update_rl(&(%s%s),&(%s%s),%d,%d,refnum,counter);",
                 CkMyData,flist,
                 CkMyData,rl,when_set->wnode->wno,when_set->wnode->wtype);
        writeoutput(temp,NOFREE); WriteReturn();
        writeoutput("dispatched = 1;}",NOFREE); WriteReturn();
        when_set = when_set->next;
      } 
      writeoutput("    break;",NOFREE); WriteReturn();
      cond = cond->next; 
    }
    writeoutput(" }",NOFREE); WriteReturn();
    sprintf(temp,"if (dispatched && (%s%s == 0) ) _dag10%s(%s);",
                 CkMyData,activator,current_dag->charename,CkLocalPtr);
    writeoutput(temp,NOFREE); WriteReturn();
    writeoutput("}",NOFREE); WriteReturn();

#ifdef STRACE
   /* for historical reasons , change the refnum to -1 if it is zero */
   /* remember, zero means no-reference number                       */
   /* however, simulator requires -1 for this part only              */
   sprintf(temp,"t2=dag_timer()-t2;_dag_s_r(&(%s%s),%s,var->index,((refnum==0)?-1:refnum),t2);",
           CkMyData,cid,charenum);
   writeoutput(temp,NOFREE); WriteReturn();
#endif

    writeoutput("}",NOFREE); WriteReturn();

#ifdef STRACE
    emit_mwset();
#endif
}


_dag2_entry_code(modulename,charename)
char *modulename,*charename;
{
    ELIST *eptr;
    char  *map_ename;
    int   init_flag,init_entry;

    if (InPass1) return;

    init_flag = init_entry = FALSE;
    eptr = current_dag->e_list;

    while (eptr) {

      map_ename = Map(modulename,charename,eptr->name); 
      entry_header_code(eptr->msgtype,map_ename,charename);
      writeoutput("_dag3_BUFFER *buffer;int dispatched;",NOFREE); WriteReturn();
      /* if init entry, emit initialization code */
      if (strcmp(eptr->name,"init") == 0) {
         if (init_flag) printf("dag: init error\n",EXIT);
         init_flag = init_entry = TRUE;
         init_code();
      }


      /* get the buffer pointer */
      if (IS_MATCHING(eptr))
          sprintf(temp,"buffer=_dag4_mfb(&(%s%s),%s%s,%d,GetRefNumber(msg));",
                  CkMyData,flist,CkMyData,epbuffer,eptr->eno);
      else
          sprintf(temp,"buffer = _dag4_nonmfb(&(%s%s),%s%s,%d);",
                  CkMyData,flist,CkMyData,epbuffer,eptr->eno);
      writeoutput(temp,NOFREE);
      WriteReturn();

      /* if a new buffer, initialize it */
      writeoutput("if(buffer->ecount == -1){",NOFREE);
      if ( IS_MULTIPLE(eptr) ) 
        sprintf(temp,"  buffer->ecount=%s%s;",CkMyData,eptr->varname);
      else
        sprintf(temp,"  buffer->ecount=1;");
      writeoutput(temp,NOFREE);
      sprintf(temp,"  buffer->free_count=%d;}",eptr->numofwhen);
      writeoutput(temp,NOFREE); WriteReturn();

      /* put the message */
      if ( IS_MULTIPLE(eptr) )
          sprintf(temp,"_dag4_mpm(buffer,msg,%s%s);",CkMyData,eptr->varname);
      else
          sprintf(temp,"_dag4_opm(buffer,msg);");
      writeoutput(temp,NOFREE);
      WriteReturn();

      if (init_entry) {
         writeoutput("buffer->expect=1;",NOFREE);
         writeoutput("buffer->refnum=0;",NOFREE);	/* Added 7/21/94 Ed K. */
			/* The above initialization gets rid of a Purify UMR error - E.K. */
         WriteReturn();
      }

      /* call the entry function */
      writeoutput("if ( (buffer->expect) )",NOFREE);
      sprintf(temp,"dispatched=_dag5_1%s(%s,%d,buffer->refnum,buffer->ecount,%d,0,buffer);",
        current_dag->charename,CkLocalPtr,eptr->eno,IS_MULTIPLE(eptr)?1:0);
      writeoutput(temp,NOFREE); WriteReturn();

      /* emit code to check ready list */
      sprintf(temp," if (dispatched) _dag4_process_rl(&(%s%s),%s,&(%s%s),%s%s,&(%s%s));",
              CkMyData,flist,
              CkLocalPtr,CkMyData,rl,CkMyData,wswitch_fptr,
              CkMyData,activator);
      writeoutput(temp,NOFREE); 
      writeoutput("}",NOFREE); WriteReturn();
  
      init_entry = FALSE; 
      eptr=eptr->next;
    }  

    emit_expect_code();
     
}

_dag2_efunction_code()
{
      EWLIST *when_set;
      ELIST  *eptr;

      if (InPass1) return;

      eptr = current_dag->e_list;

      /* function header */
      sprintf(temp,"static int _dag5_1%s(%s,index,refnum,ecount,msgcount,from_expect,buffer)",
              current_dag->charename,CkLocalPtr);
      writeoutput(temp,NOFREE);WriteReturn();
      declare_lptr();
      writeoutput("int index,refnum,ecount,msgcount,from_expect;",NOFREE); 
      WriteReturn();
      writeoutput("_dag3_BUFFER *buffer;",NOFREE); WriteReturn();
      writeoutput("{int dispatched;_dag3_COUNT *counter;",NOFREE);
      WriteReturn(); 
      assign_lptr();

      writeoutput("dispatched=0;",NOFREE); WriteReturn();


      writeoutput("switch (index) {",NOFREE); WriteReturn();

      while(eptr) {

         when_set = eptr->wlist;
         sprintf(temp,"   case %d:",eptr->eno);
         writeoutput(temp,NOFREE); WriteReturn();

         while(when_set) {

           /* if it is a when of type ANY, don't create a count node */
           /* just dispatch it, since only one ANY condition allowed */
           
           if ( IS_ANY(when_set->wnode) )
            sprintf(temp,"if (msgcount > 0) dispatched|=_dag4_cci(&(%s%s),%d,%d,&(%s%s),msgcount,refnum,buffer);",
                    CkMyData,flist,when_set->wnode->wno,when_set->wnode->wtype,
                    CkMyData,rl);
           else {
            writeoutput("if (ecount==0) {",NOFREE); WriteReturn();
            writeoutput("dispatched |= ",NOFREE); 
            if ( IS_MATCHING_WHEN(when_set->wnode) )
              sprintf(temp,"_dag4_ccm(&(%s%s),%d,%s%s,%s%s,%d,%d,&(%s%s),refnum,buffer);",
                      CkMyData,flist,when_set->position,
                      CkMyData,wcount,CkMyData,wcounter,
                      when_set->wnode->wno,when_set->wnode->wtype,
                      CkMyData,rl);
            else
              sprintf(temp,"_dag4_ccn(&(%s%s),%d,%s%s,%s%s,%d,%d,&(%s%s),refnum,buffer);",
                      CkMyData,flist,when_set->position,
                      CkMyData,wcount,CkMyData,wcounter,
                      when_set->wnode->wno,when_set->wnode->wtype,
                      CkMyData,rl);
            writeoutput(temp,NOFREE); WriteReturn();

            if (IS_MULTIPLE(eptr)) {
              sprintf(temp,"buffer->ecount = %d;}",MULTIPLE);
              writeoutput(temp,NOFREE);
              }
            else
              writeoutput("}",NOFREE);
            WriteReturn();
           }

           when_set = when_set->next;
         }
         writeoutput(" break;",NOFREE); WriteReturn();
         eptr = eptr->next;
      }

      writeoutput("}",NOFREE); WriteReturn();
      sprintf(temp,"if (from_expect && (%s%s == 0) && dispatched )",
              CkMyData,activator);
      writeoutput(temp,NOFREE);
      sprintf(temp,"_dag10%s(%s);", current_dag->charename,CkLocalPtr); 
      writeoutput(temp,NOFREE); WriteReturn();
      writeoutput("return dispatched;}",NOFREE); WriteReturn();
}





_dag2_conv0()
{
     if (InPass1) return;
     sprintf(temp,"static _dag11%s(%s,epconv) int epconv[];",
             current_dag->charename,CkLocalPtr);
     writeoutput(temp,NOFREE); WriteReturn();
     declare_lptr();
     writeoutput("{",NOFREE); 
     assign_lptr();
     savedeptr = current_dag->e_list;
}


_dag2_conv1()
{
    if (InPass1) return;
    writeoutput("}",NOFREE); WriteReturn();
    /* emit_epconv(); */
}

_dag2_conv2()
{
    if (InPass1) return;
    sprintf(temp,"%s%s[%d] = ",CkMyData,epconv,savedeptr->eno);
    writeoutput(temp,NOFREE);
}

_dag2_conv3()
{
    if (InPass1) return;
    writeoutput(";",NOFREE);
    savedeptr = savedeptr->next;
}


static emit_expect_code()
{


    sprintf(temp,"static _dag5%s(%s,ep,refnum)",
               current_dag->charename,CkLocalPtr);
    writeoutput(temp,NOFREE);WriteReturn();
    declare_lptr();
    writeoutput("int ep,refnum;{int index,ecount,msgcount;_dag3_BUFFER *buffer;",NOFREE);
#ifdef STRACE
writeoutput("unsigned int t2;",NOFREE);
#endif
    WriteReturn();
    assign_lptr();

#ifdef STRACE
    writeoutput("t2 = dag_timer();",NOFREE); WriteReturn();
#endif

    sprintf(temp,"index = _dag4_epconv(ep,%s%s,%d);",
        CkMyData,epconv,current_dag->numofentry);
    writeoutput(temp,NOFREE);WriteReturn();


    sprintf(temp,"buffer=_dag4_fb(&(%s%s),%s%s,index,%s%s[index],refnum,&msgcount);",
        CkMyData,flist,CkMyData,epbuffer,CkMyData,etype);
    writeoutput(temp,NOFREE); WriteReturn();

    sprintf(temp,"_dag5_1%s(%s,index,refnum,buffer->ecount,msgcount,1,buffer);",
        current_dag->charename,CkLocalPtr); WriteReturn();
    writeoutput(temp,NOFREE); WriteReturn();

#ifdef STRACE
   sprintf(temp,"t2=dag_timer()-t2;_dag_s_e(&(%s%s),%s,ep,refnum,t2);",
           CkMyData,cid,charenum);
   writeoutput(temp,NOFREE); WriteReturn();
#endif

   writeoutput("}",NOFREE); WriteReturn();




}

#ifdef STRACE 
static emit_mwset()
{
    ELIST *eptr;
    EWLIST *ewptr;

    if (InPass1) return;
    sprintf(temp,"static _dag8%s(%s)",current_dag->charename,CkLocalPtr);
    writeoutput(temp,NOFREE);WriteReturn();
    declare_lptr();
    writeoutput("{",NOFREE); WriteReturn();
    assign_lptr();
    eptr = current_dag->e_list; 
    while(eptr) {
       if (IS_MULTIPLE(eptr) ) {
          ewptr = eptr->wlist;
          while(ewptr) {
          /*   if (!IS_ANY(ewptr->wnode)) { */
             
                sprintf(temp,"_dag_s_mset(%d,%s%s);",ewptr->wnode->wno,
                              CkMyData,eptr->varname);
                writeoutput(temp,NOFREE); WriteReturn();
         /*    } */
             ewptr = ewptr->next;
          }
       }
       eptr = eptr->next;
    }
    writeoutput("}",NOFREE); WriteReturn();
} 


#endif




/*
static emit_epconv()
{
    if (InPass1) return;
    sprintf(temp,"static _dag8%s(ep,epconv,n) int ep,epconv[],n;",
                  current_dag->charename);
    writeoutput(temp,NOFREE); WriteReturn();
    writeoutput("{int i; for(i=0;i<n;i++) if(epconv[i] == ep)return i;",NOFREE);
    WriteReturn();
    writeoutput("CmiPrintf(\"dag: invalid entry point\\n\");}",NOFREE);
    WriteReturn();
}
*/


_dag2_send_act(i)
int i;
{
     if (InPass1) return;
     switch(i) {
        case 0: /* begininning part */
            sprintf(temp,"static _dag10%s(%s)",
                    current_dag->charename,CkLocalPtr);
            writeoutput(temp,NOFREE); WriteReturn();
            declare_lptr();
            writeoutput("{_dag3_MSG *msg;",NOFREE); WriteReturn();
            assign_lptr();
            writeoutput(" msg = (_dag3_MSG *) GenericCkAlloc(",NOFREE);
            break;
        case 1:
            writeoutput(",sizeof(_dag3_MSG),0);",NOFREE); WriteReturn(); 
            if (current_dag->is_a_chare) 
               writeoutput("SendMsg(",NOFREE);
            else
            writeoutput("_CK_SendMsgBranch(",NOFREE);
            break;
        case 2:
            if (current_dag->is_a_chare)
                sprintf(temp,",msg,&(%s%s));",CkMyData,cid);
            else {
                writeoutput(",msg,_CK_MyBocNum(_CK_4mydata),",NOFREE);
                sprintf(temp,"%s%s);",CkMyData,penum);
            }
            writeoutput(temp,NOFREE); WriteReturn();
            sprintf(temp,"%s%s = 1;}",CkMyData,activator);
            writeoutput(temp,NOFREE); WriteReturn();
     } 
}





_dag2_activator(modulename,charename)
char *modulename,*charename;
{
    char *map_ename;
    if (InPass1) return ;
    map_ename = Map(modulename,charename,"_dag9"); 
    entry_header_code("_dag3_MSG",map_ename,charename);

    sprintf(temp,"%s%s = 0;",CkMyData,activator);
    writeoutput(temp,NOFREE); WriteReturn();
    sprintf(temp,"_dag4_process_rl(&(%s%s),%s,&(%s%s),%s%s,&(%s%s));",
            CkMyData,flist,
            CkLocalPtr,CkMyData,rl,CkMyData,wswitch_fptr,
            CkMyData,activator);
    writeoutput(temp,NOFREE);WriteReturn();
    writeoutput("CkFreeMsg(msg);}",NOFREE); WriteReturn();
}


_dag2_set_current_when()
{
    if (InPass1) return;
    if (current_when == NULL) 
        current_when = current_dag->w_list;
    else
        current_when = current_when->next;
}

    

_dag2_when_header(CurrentTable)
void *CurrentTable;
{
    WLIST *wptr;
    WCONDLIST *ent_cond;

    if (InPass1) return;

    wptr = current_when;

      /* write when function */
      sprintf(temp,"static  _dag6%s%d(%s",
              current_dag->charename,wptr->wno,CkLocalPtr);
      writeoutput(temp,NOFREE);
      ent_cond = wptr->clist;
      while (ent_cond) {
          if (ent_cond->isentry) {
             writeoutput(",",NOFREE);
             writeoutput(((ELIST *)ent_cond->eptr)->msgname,NOFREE); 
          }
          ent_cond = ent_cond->next;
      }
      writeoutput(") ",NOFREE);
      WriteReturn();
      ent_cond = wptr->clist;
      declare_lptr();
      while(ent_cond) {
         if(ent_cond->isentry) {
             sprintf(temp,"%s *%s",
                 ((ELIST*)ent_cond->eptr)->msgtype,
                 ((ELIST*)ent_cond->eptr)->msgname);
             writeoutput(temp,NOFREE);
             Insert(((ELIST*)ent_cond->eptr)->msgname,CurrentTable);
             if (  IS_MULTIPLE(((ELIST*)ent_cond->eptr))  &&
                   !IS_ANY(wptr) )
                writeoutput("[]",NOFREE); 
             writeoutput(";",NOFREE); WriteReturn();
         }
         ent_cond = ent_cond->next;
      }
      

}


_dag2_whenswitch()
{
     WLIST *wptr;
     WCONDLIST *ent_cond;
     int i,j,flag,numofp;

     if (InPass1) return;

     /* write the switch function for when blocks */
     wptr = current_dag->w_list;
     sprintf(temp,"static _dag4%s(%s,rlnode)",
             current_dag->charename,CkLocalPtr);
     writeoutput(temp,NOFREE); WriteReturn();
     declare_lptr();
     writeoutput("_dag3_RLNODE *rlnode; ",NOFREE); WriteReturn();
     writeoutput("{ ",NOFREE); WriteReturn();
     assign_lptr();
     writeoutput(" switch (rlnode->value) {",NOFREE); WriteReturn();
     while(wptr) {

        ent_cond = wptr->clist; i = 0;

        if (!ent_cond) error("Empty when condition list",EXIT);
        sprintf(temp,"   case %d :{",wptr->wno);
        writeoutput(temp,NOFREE); WriteReturn();
     
        while(ent_cond) {
           if (ent_cond->isentry) i++; 
           ent_cond = ent_cond->next;
        }

        if ( numofp=i > _dag3_WLIMIT ) 
            error("dag: too many entries in a when condition list. Increase _dag3_WLIMIT\n",EXIT);

        /* declare parameters */
        if (i) {


           writeoutput("void *p0",NOFREE);
           for(j=1; j<i; j++) {
              sprintf(temp,",*p%d",j);
              writeoutput(temp,NOFREE);
           }
           writeoutput(";",NOFREE); WriteReturn();
     
          
           /* assign parameters */ 
           ent_cond = wptr->clist; i=0;
           while(ent_cond) {
              if (ent_cond->isentry) {
                  flag = 0;
                  sprintf(temp,"p%d=_dag4_gb(rlnode,%d,%d,",i++,
                          ent_cond->position, 
                          ((ELIST *)ent_cond->eptr)->eno); 
                  writeoutput(temp,NOFREE);
                  if ( IS_MULTIPLE(((ELIST *)ent_cond->eptr)) )flag |= MULTIPLE;
                  if ( IS_ANY(wptr) ) flag |= ANY;
                  sprintf(temp,"%d);",flag);
                  writeoutput(temp,NOFREE);
                  WriteReturn();
              }
              ent_cond = ent_cond->next;
           }
           WriteReturn();  
        }


#ifdef STRACE
   sprintf(temp,"_dag_s_wb(&(%s%s),%s,%d,rlnode->refnum);",
           CkMyData,cid,charenum,wptr->wno);
   writeoutput(temp,NOFREE); WriteReturn();
#endif
        /* call the when function */
        ent_cond = wptr->clist; i = 0;
        /* first, set the current_refnum var */
        sprintf(temp,"%s%s = rlnode->refnum\n;",CkMyData,current_refnum);
        writeoutput(temp,NOFREE); WriteReturn(); 
        sprintf(temp,"_dag6%s%d(%s",
                current_dag->charename,wptr->wno,CkLocalPtr);
        writeoutput(temp,NOFREE);
        while(ent_cond) {
           if (ent_cond->isentry) {
               sprintf(temp,",p%d",i++);
               writeoutput(temp,NOFREE);
           }
           ent_cond = ent_cond->next;
        }
        writeoutput(");",NOFREE);

#ifdef STRACE
   writeoutput("_dag_s_we();",NOFREE); WriteReturn();
#endif

        /* free the buffer and messages if necessary */

        if ( !IS_ANY(wptr)  )  {
           /* code to free the buffers. get_buffer decrements
              the free_count (except when stmts of ANY type 
              if free_count is zero, that buffer and messsages
              can be freed */
       
           writeoutput("_dag4_freebuffer",NOFREE);
           sprintf(temp,"(&(%s%s),rlnode);",CkMyData,flist);
           writeoutput(temp,NOFREE);WriteReturn();

/*
           ent_cond = wptr->clist;
           while(ent_cond) {
             if (ent_cond->isentry) {
                if ( IS_MULTIPLE(((ELIST *)ent_cond->eptr)) )
                   writeoutput("_dag4_m_freebuffer",NOFREE);
                else
                   writeoutput("_dag4_freebuffer",NOFREE);
                sprintf(temp,"(&(%s%s),rlnode);",CkMyData,flist);
                writeoutput(temp,NOFREE);WriteReturn();
             }
             ent_cond = ent_cond->next;  
           }
*/
        }
            
           
        writeoutput("} break;",NOFREE); WriteReturn();
    
        wptr = wptr->next;
     }

     writeoutput("} }",NOFREE);WriteReturn();
}




static reverse_elist()
{
    ELIST *prev,*current,*next;

    if (current_dag->e_list == (ELIST *) NULL) return;
    if (current_dag->e_list->next == (ELIST *) NULL) return;


    prev = (ELIST *) NULL;
    current = current_dag->e_list;
    while(current) {
       next = current->next;
       current->next = prev;
       prev = current;
       current = next;
    }
    current_dag->e_list = prev; 
}   


static reverse_clist()
{
    CLIST *prev,*current,*next;

    if (current_dag->c_list == (CLIST *) NULL) return;
    if (current_dag->c_list->next == (CLIST *) NULL) return;
    

    prev = (CLIST *) NULL;
    current = current_dag->c_list;
    while(current) {
       next = current->next;
       current->next = prev;
       prev = current;
       current = next;
    }
    current_dag->c_list = prev;
}

        
       
static ELIST *is_entry(condname)
char *condname;
{
     ELIST *ptr;
     ptr = e_list;
     while(ptr!= NULL) 
        if(strcmp(condname,ptr->name) == 0) 
          return ptr;
        else
          ptr = ptr->next;
     return NULL;
}

static CLIST *insert_cond(condname)
char *condname;
{
     CLIST *ptr,*cond;
     ptr = c_list;
     while(ptr != NULL)
        if(strcmp(condname,ptr->name) == 0) 
           return ptr;
        else
           ptr = ptr->next;
     cond = (CLIST *) malloc(sizeof(CLIST));
     cond->wlist = NULL;
     cond->cno = current_cno++;
     cond->next = c_list;
     c_list = cond;
     c_list->name = (char *) malloc(strlen(condname)+1);
     strcpy(c_list->name,condname);
     return cond;
}






static init_code()
{
   ELIST *eptr;
   WLIST *wptr;
   CLIST *cond;
   int flag;


    sprintf(temp,"int _dag4%s(),_dag5%s(),_dag7%s();",
         current_dag->charename,current_dag->charename,current_dag->charename);
    writeoutput(temp,NOFREE); WriteReturn();

#ifdef STRACE
    sprintf(temp,"int _dag8%s();",current_dag->charename);
    writeoutput(temp,NOFREE); WriteReturn();
#endif

   /* emit code to initiaalize condtrol data */
   sprintf(temp,"%s%s.head = %s%s.tail = (_dag3_RLNODE *) NULL;",
           CkMyData,rl,CkMyData,rl);   
   writeoutput(temp,NOFREE); WriteReturn();

   
   eptr = current_dag->e_list;
   while(eptr) {
      sprintf(temp,"%s%s[%d] = %d;",CkMyData,etype,eptr->eno,eptr->etype);
      writeoutput(temp,NOFREE); WriteReturn();
      sprintf(temp,"%s%s[%d] = (_dag3_BUFFER *) NULL;",
              CkMyData,epbuffer,eptr->eno);
      writeoutput(temp,NOFREE); WriteReturn();
      eptr = eptr->next;
   }

   wptr = current_dag->w_list;
   while(wptr) {
      sprintf(temp,"%s%s[%d] = %d;",CkMyData,wcount,wptr->wno,wptr->count);
      writeoutput(temp,NOFREE); WriteReturn();
      sprintf(temp,"%s%s[%d] = (_dag3_COUNT *) NULL;",
              CkMyData,wcounter,wptr->wno);
      writeoutput(temp,NOFREE); WriteReturn();
      wptr = wptr->next;
   } 

   cond = current_dag->c_list;
   while(cond) {
      sprintf(temp,"%s%s.index = %d;",CkMyData,cond->name,cond->cno);
      writeoutput(temp,NOFREE); WriteReturn();
      cond = cond->next;
   }

   sprintf(temp,"%s%s = _dag7%s;",
           CkMyData,condvar_fptr,current_dag->charename);
   writeoutput(temp,NOFREE); WriteReturn();
#ifdef STRACE
   sprintf(temp,"%s%s = _dag8%s;",
           CkMyData,mw_fptr,current_dag->charename);
   writeoutput(temp,NOFREE); WriteReturn();
#endif 
   sprintf(temp,"%s%s = _dag5%s;",CkMyData,efunc_fptr,current_dag->charename);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s=_dag4%s;",CkMyData,wswitch_fptr,current_dag->charename);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s = 0;",CkMyData,activator);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s = %d;",CkMyData,ischare,current_dag->is_a_chare);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s.dagexit = 0;",CkMyData,rl);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s.bcount = %s%s.ccount = 0;",
                CkMyData,flist,CkMyData,flist,CkMyData,flist);
   writeoutput(temp,NOFREE); WriteReturn();
   sprintf(temp,"%s%s.b = 0;%s%s.c = 0;",
                CkMyData,flist,CkMyData,flist,CkMyData,flist);
   writeoutput(temp,NOFREE); WriteReturn();
   if (current_dag->is_a_chare) 
      sprintf(temp,"MyChareID(&(%s%s));",CkMyData,cid);
   else {
       sprintf(temp,"_CK_MyBranchID(&(%s%s),%s);",CkMyData,cid,CkLocalPtr);
       writeoutput(temp,NOFREE); WriteReturn();
       sprintf(temp,"%s%s = McMyPeNum();",CkMyData,penum);
   }
   writeoutput(temp,NOFREE); WriteReturn();
   
   sprintf(temp,"_dag11%s(%s,%s%s);",
           current_dag->charename,CkLocalPtr,CkMyData,epconv);
   writeoutput(temp,NOFREE);
   sprintf(temp,"%s%s = %d;",CkMyData,current_refnum,DAG_NOREF);
   writeoutput(temp,NOFREE); WriteReturn();

#ifdef STRACE
   WriteReturn();
   sprintf(temp,"_dag_s_init(&(%s%s),%s,%d,%s%s,\"%s\");",
           CkMyData,cid,charenum,current_dag->numofentry,CkMyData,epconv,
           trace_charename);
   writeoutput(temp,NOFREE); WriteReturn();

#endif

}






static entry_header_code(msgtype,map_ename,charename)
char  *map_ename,*charename,*msgtype;
{
      char *ename; 
    
      writeoutput("void static ",NOFREE);
      writeoutput(map_ename,NOFREE);
      writeoutput("(msg,",NOFREE);
      writeoutput(CkLocalPtr,NOFREE);
      writeoutput(")",NOFREE);
      WriteReturn();
      declare_lptr();
      writeoutput(msgtype,NOFREE);
      writeoutput(" *msg;",NOFREE);
      WriteReturn();
      writeoutput("{ ",NOFREE);
      assign_lptr();
}






static declare_lptr()
{
      writeoutput("void *",NOFREE);
      writeoutput(CkLocalPtr,NOFREE);
      writeoutput(";",NOFREE);
      WriteReturn();
}


static assign_lptr()
{
      writeoutput(current_dag->charename,NOFREE);
      writeoutput(DataSuffix,NOFREE);
      writeoutput(AssignMyDataPtr,NOFREE);
      writeoutput(current_dag->charename,NOFREE);
      writeoutput(DataSuffix,NOFREE);
      writeoutput(" *)",NOFREE);
      writeoutput(CkLocalPtr,NOFREE);
      writeoutput(";",NOFREE);
      WriteReturn();
}

static fill_charenum(chare)
char  *chare;
{
     extern char *CkPrefix_, *Prefix();
     charenum = Prefix(chare,modulename,CkPrefix_);
     sprintf(trace_charename,"%s %s",modulename,chare);
}

#ifdef STRACE

static write_depn()
{
    ELIST     *eptr;
    CLIST     *cptr;
    WLIST     *wptr;
    EWLIST    *weptr;
    WCONDLIST *clist;
    int       count,count2;
   
    FILE *fp;

    if ( (fp = fopen("dag_depn","a")) == NULL ) {
       printf("can't open dag_depn for append\n");
       return; 
    }

    fprintf(fp,"b %s %s ",modulename,current_dag->charename);
    eptr = current_dag->e_list;
    for(count=0; (eptr) ; eptr=eptr->next) count++;
    fprintf(fp,"%d ",count);
    cptr = current_dag->c_list;
    for(count=0; (cptr) ; cptr=cptr->next) count++;
    fprintf(fp,"%d ",count);
    wptr = current_dag->w_list;
    for(count=0; (wptr) ; wptr=wptr->next) count++;
    fprintf(fp,"%d\n",count);

    eptr = current_dag->e_list;
    while (eptr) {
       fprintf(fp,"e %d ",eptr->eno); 
       weptr = eptr->wlist;
       for(count=0; (weptr) ; weptr=weptr->next) count++;
       fprintf(fp,"%d\n",count); 
       weptr = eptr->wlist;
       while (weptr) {
          fprintf(fp,"%d\n",weptr->wnode->wno);
          weptr = weptr->next;
       }
       eptr = eptr->next;      
    }

    cptr = current_dag->c_list;
    while (cptr) {
       fprintf(fp,"c %d ",cptr->cno);
       weptr = cptr->wlist;
       for(count = 0; (weptr) ; weptr=weptr->next) count++;
       fprintf(fp,"%d\n",count);
       weptr = cptr->wlist;
       while (weptr) {
           fprintf(fp,"%d\n",weptr->wnode->wno);
           weptr=weptr->next;
       }
       cptr = cptr->next;
    }

    wptr = current_dag->w_list;
    while (wptr) {
       fprintf(fp,"w %d ",wptr->wno);
       if (IS_ANY(wptr) )  {
           count = 1;
           count2 = -1;
          }
       else {
          clist = wptr->clist;
          count = 0;
          count2 = 0;
          while (clist) {
             if (clist->isentry) {
                  count ++;
                  if ( IS_MULTIPLE(((ELIST *)clist->eptr)) ) 
                     count2++;
                  else
                     if (strcmp(((ELIST *)clist->eptr)->name,"init")) count ++;
                }
             else 
                count++;
             clist = clist->next;
         }
      }
      fprintf(fp,"%d %d\n",count,count2);
      if (count2>0) {
          clist = wptr->clist;
          while(clist) {
              if (clist->isentry)
                     if ( IS_MULTIPLE(((ELIST *)clist->eptr)) ) 
                     fprintf(fp,"%d\n",((ELIST *)clist->eptr)->eno);
              clist = clist->next;
          }
      }
      wptr = wptr->next; 
    }

    fclose(fp);
} 
#endif
 
