/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 1.1  1995-06-13 11:32:16  jyelon
 * Initial revision
 *
 * Revision 1.1  1995/06/13  10:06:34  jyelon
 * Initial revision
 *
 * Revision 1.2  1994/11/11  05:20:08  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:37:46  brunner
 * Initial revision
 *
 ***************************************************************************/
/* _dag3_WLIMIT is defien d in dag.c dagger..h also */
#define _dag3_WLIMIT 8
#define _dag3_FREEBLIMIT 4
#define _dag3_FREECLIMIT 4
#define _dag3_FREERLIMIT 4

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
    struct s_dag3_BUFFER *bpa[_dag3_WLIMIT];
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
