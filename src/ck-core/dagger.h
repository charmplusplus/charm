#ifndef DAGGER_H
#define DAGGER_H

/* _dag3_WLIMIT is defined in dag.c dag.h also */
#define _dag3_WLIMIT 8

struct s_dag3_DAGVAR {
    int index;
    int init_value;
    int counter;
};

struct s_dag3_RLNODE {
    int    wno;
    int    refnum;
    struct s_dag3_RLNODE *next;
};


struct s_dag3_BUFFER {
    int eno;
    int refnum;
    int expect;
    int ecount;
    int free_count;
    void *msg;
    struct s_dag3_BUFFER **prev;
    struct s_dag3_BUFFER *next;
};


struct s_dag3_COUNT {
    int refnum;
    int value;
    int bix;
    struct s_dag3_BUFFER *bpa[_dag3_WLIMIT]; /* back pointer array */
    struct s_dag3_COUNT  **prev;
    struct s_dag3_COUNT  *next;
};

struct s_dag3_FREELIST {
    int bcount;
    int ccount;
    struct s_dag3_BUFFER *b;
    struct s_dag3_COUNT  *c;
};

struct s_dag3_RL {
    int    dagexit;
    struct s_dag3_COUNT *head;
    struct s_dag3_COUNT *tail;
};

typedef struct s_dag3_COUNT _dag3_RLNODE;
typedef struct s_dag3_RL     _dag3_RL;
typedef struct s_dag3_BUFFER _dag3_BUFFER;
typedef struct s_dag3_COUNT  _dag3_COUNT;
typedef struct s_dag3_FREELIST _dag3_FREELIST;
typedef struct s_dag3_DAGVAR _dag3_DAGVAR;
typedef struct s_dag3_DAGVAR DAGVAR;
typedef struct s_dag3_DAGVAR CONDVAR;

extern _dag3_COUNT  *_dag4_mfc();
extern _dag3_COUNT  *_dag4_nonmfc();
extern _dag3_BUFFER *_dag4_mfb();
extern _dag3_BUFFER *_dag4_nonmfb();
extern _dag3_BUFFER *_dag4_fb();
extern void         *_dag4_gb();

/* DAG_NOREF define in dag.c also */
#define DAG_NOREF 0
#define expect(x,y) (*_CK_4mydata->_dag3_epf_fptr)(_CK_4mydata,x,y)
#define dagExit()   (_CK_4mydata->_dag3_rl.dagexit = 1)
#define eset(n,v)   (n=v)
#define set(c,v)    (c.counter=v)
#define inq(c)      (c.counter)
#define decrement(c) (*_CK_4mydata->_dag3_cv_fptr)(_CK_4mydata,&(c),0,DAG_NOREF)
#define ready(c,r) (*_CK_4mydata->_dag3_cv_fptr)(_CK_4mydata,&(c),(r==DAG_NOREF)?0:1,(r==DAG_NOREF)?((c.counter=1)-1):r)
#define GetMyRefNumber() _CK_4mydata->_dag3_myrefnum
message {int i;} _dag3_MSG;

#ifdef STRACE
/* trace function calls */
extern _dag_s_sbranch1();
extern _dag_s_sbranch2();
extern _dag_s_broadcast();
extern _dag_s_smsg();
extern _dag_s_msetcond();
extern _dag_s_endsend();


#define dag_sendbranch1(e,m,p)    _dag_s_sbranch1(&_CK_4mydata->_dag3_cid,p,e,m)
#define dag_sendbranch2(e,m,p,b)  _dag_s_sbranch2(b,p,e,m)
#define dag_broadcast(e,m)        _dag_s_broadcast(&_CK_4mydata->_dag3_cid,e,m)
#define dag_send(e,m,c)           _dag_s_smsg(c,e,m)
#define dag_mwset()               (*_CK_4mydata->_dag3_mw_fptr)(_CK_4mydata)
#define dag_mcset(c,v)            _dag_s_msetcond(c.index,v)
#define dag_endsend()             _dag_s_endsend();

#else

#define dag_sendbranch1(e,m,p)
#define dag_sendbranch2(e,m,p,b)  
#define dag_broadcast(e,m)
#define dag_send(e,m,c)
#define dag_mwset()   
#define dag_mcset(c,v) 
#define dag_endsend()  

#endif


#endif
