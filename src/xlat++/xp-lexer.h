#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define TRUE 1
#define FALSE 0

#define MAXAGGS 1024
#define MAXCHARES 256
#define MAXMSGS 256
#define MAXACCS 256
#define MAXDTABLES 256
#define MAXREADS 512
#define MAXSYMBOLS 1024
#define MAX_NAME_LENGTH 256
#define MAX_OUTBUF_SIZE 1024
#define MAX_PARENTS 128
#define MAXIDENTS 256
#define MAX_MODULES 256
#define MAX_VARSIZE 64
#define MAX_FUNCTIONS 256
#define MAX_TOKEN_SIZE 1024

/* Types of Sends */
#define BROADCAST 123
#define SIMPLE 124
#define BRANCH 125

#define READMSG 1091

#define DTABLE 54321

#define BASE_PERM_INDEX 0x00008001

#define FLUSHBUF() fprintf(outfile,"%s",OutBuf) ; strcpy(OutBuf,"") ; \
                        fflush(outfile);

struct chareinfo ;

typedef struct ep {
        char epname[MAX_NAME_LENGTH] ;
        char msgname[MAX_NAME_LENGTH] ;
        int inherited ;
        int isvirtual ;
	int defined ;  /* defined (code present) in this module or not */
	struct ep * parentep ;
	struct chareinfo * chare ;	/* chare in which this ep is */

        struct ep *next ;
} EP ;

typedef struct fn {
	char fnname[MAX_NAME_LENGTH] ;
	struct fn *next ;
} FN ;

struct charelist ;

typedef struct chareinfo {
        char name[256] ;
        EP *eps ;
	FN *fns ;
        struct charelist *parents ;
} ChareInfo ;

typedef struct charelist {
        ChareInfo *chare ;
        struct charelist *next ;
} ChareList ;

typedef struct accstruct {
	char *name ;
	char *initmsgtype ;
	char *msgtype ;
	char *msg ;
	int defined ;
} AccStruct ;

typedef struct varsizestruct {
	char *name ;
	char *type ;
} VarSizeStruct ;

typedef struct msgstruct {
        char *name ;    /* name of a msg */
	int pack ; /* TRUE if this msg has a pack/unpack */
	int numvarsize ; /* number of varsize fields in this message */
	VarSizeStruct *varsizearray ; /* array of varSize fields' info */
} MsgStruct ;

typedef struct handle_entry {
        char *name ;    /* name of a handle identifier */
/*      int nameindex ;  its type : index into Chare/BOC/Acc/... table */
	char *typestr ;
} HandleEntry ;

typedef struct sym_entry {
        char *name ;    /* name of a typedef */
        int permanentindex ; /* its index into PermanentAggTable, if agg */
} SymEntry ;

typedef struct functionstruct {	/* global functions */
	char *name ;
	int defined ;
} FunctionStruct ;

typedef struct stackstruct { 
/* This structure is the nested scope stack element. It
   holds a record of the all tables, and is used to
   modify these tables when entering or leaving scoping constructs.  */
	int charecount ;
	int boccount ;
	int TotalMsgs ;
	int TotalAccs ;
	int TotalMonos ;
	int TotalReads ;
	int TotalReadMsgs ;
	int TotalSyms ;
	int ChareHandleTableSize ;
	int BOCHandleTableSize ;
	int AccHandleTableSize ;
	int MonoHandleTableSize ;

	struct stackstruct *next ;
} StackStruct ;


typedef struct aggstate {
/* This structure holds the state of all tables :
   the objects defined INSIDE an aggregate type.
   It is used to modify the tables when say, entering the
   body of a Agg::function defined outside the aggregate.	*/

	char name[MAX_NAME_LENGTH] ;

	ChareInfo ** ChareTable ;
	ChareInfo ** BOCTable ;
	MsgStruct * MessageTable ;
	AccStruct ** AccTable ;
	AccStruct ** MonoTable ;
	char ** ReadTable ;
	char ** ReadMsgTable ;
	SymEntry *SymTable ;
	HandleEntry *ChareHandleTable ;
	HandleEntry *BOCHandleTable ;
	HandleEntry *AccHandleTable ;
	HandleEntry *MonoHandleTable ;

	int charecount ;
	int boccount ;
	int TotalMsgs ;
	int TotalAccs ;
	int TotalMonos ;
	int TotalReads ;
	int TotalReadMsgs ;
	int TotalSyms ;
	int ChareHandleTableSize ;
	int BOCHandleTableSize ;
	int AccHandleTableSize ;
	int MonoHandleTableSize ;
} AggState ;

