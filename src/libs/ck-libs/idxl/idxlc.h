/**
 * IDXL--Index List communication library.
 * C Header file.
 */
#ifndef _CHARM_IDXL_C_H
#define _CHARM_IDXL_C_H

#ifdef __cplusplus
  extern "C" {
#endif

/** Initialize the IDXL library.  Must have already called MPI_Init. */
void IDXL_Init(int comm);

/** An index list, the fundamental datatype of this library. */
typedef int IDXL_t;
#define IDXL_DYNAMIC_IDXL_T 1540000000
#define IDXL_STATIC_IDXL_T 1550000000
#define IDXL_LAST_IDXL_T 1560000000

/** Create a new, empty index list. Must eventually call IDXL_Destroy on this list. */
IDXL_t IDXL_Create(void);

/** Print the send and recv indices in this communications list: */
void IDXL_Print(IDXL_t l);

/** Copy the indices in src into l. */
void IDXL_Copy(IDXL_t l,IDXL_t src);

/** Shift the indices of this list by this amount. */
void IDXL_Shift(IDXL_t l,int startSend,int startRecv);

/** Add these indices into our list.  Any duplicates will get listed twice.
 * @param l the list to add indices to.
 * @param src the list of indices to read from and add.
 * @param startSend New first index to send from.
 * @param startRecv New first index to recv values into.
 */
void IDXL_Combine(IDXL_t l,IDXL_t src,int startSend,int startRecv);

/** Add the intersection of the entities listed in between
 * to the new entity at newIdx. */
void IDXL_Add_entity(IDXL_t l, int newIdx, int nBetween,int *between);

/** Sort the indices in this list by these 2D coordinates */
void IDXL_Sort_2d(IDXL_t l,double *coord2d);
/** Sort the indices in this list by these 3D coordinates */
void IDXL_Sort_3d(IDXL_t l,double *coord3d);

/** Throw away this index list */
void IDXL_Destroy(IDXL_t l);


/** Extract the indices out of this index list:
 */
typedef int IDXL_Side_t;
#define IDXL_SHIFT_SIDE_T_SEND 100000000 /* 16x0000000 */
#define IDXL_SHIFT_SIDE_T_RECV 200000000 /* 17x0000000 */
IDXL_Side_t IDXL_Get_send(IDXL_t l);
IDXL_Side_t IDXL_Get_recv(IDXL_t l);
int IDXL_Get_partners(IDXL_Side_t s);
int IDXL_Get_partner(IDXL_Side_t s,int partnerNo);
int IDXL_Get_count(IDXL_Side_t s,int partnerNo);
void IDXL_Get_list(IDXL_Side_t s,int partnerNo,int *list);
int IDXL_Get_index(IDXL_Side_t s,int partnerNo,int listIndex);
void IDXL_Get_end(IDXL_Side_t l);

/** Return the chunk this (ghost) local number is received from */
int IDXL_Get_source(IDXL_t l,int localNo);


/* Messaging */

/* datatypes: keep in sync with fem and idxlf.h */
#define IDXL_FIRST_DATATYPE 1510000000 /*first valid IDXL datatype*/
#define IDXL_BYTE   (IDXL_FIRST_DATATYPE+0)
#define IDXL_INT    (IDXL_FIRST_DATATYPE+1) 
#define IDXL_REAL   (IDXL_FIRST_DATATYPE+2)
#define IDXL_FLOAT IDXL_REAL /*alias*/
#define IDXL_DOUBLE (IDXL_FIRST_DATATYPE+3)
#define IDXL_INDEX_0 (IDXL_FIRST_DATATYPE+4) /*zero-based integer (c-style indexing) */
#define IDXL_INDEX_1 (IDXL_FIRST_DATATYPE+5) /*one-based integer (Fortran-style indexing) */
#define IDXL_LONG_DOUBLE (IDXL_FIRST_DATATYPE+6)

/** An IDXL_Layout_t describes the in-memory layout of a user data array */
typedef int IDXL_Layout_t;
#define IDXL_FIRST_IDXL_LAYOUT_T 1560000000

IDXL_Layout_t IDXL_Layout_create(int type,int width);
IDXL_Layout_t IDXL_Layout_offset(
	int type, int width, int offsetBytes, int distanceBytes,int skewBytes);

int IDXL_Get_layout_type(IDXL_Layout_t l);
int IDXL_Get_layout_width(IDXL_Layout_t l);
int IDXL_Get_layout_distance(IDXL_Layout_t l);

void IDXL_Layout_destroy(IDXL_Layout_t l);

/** IDXL_Comm is the transient representation for an in-progress message exchange.*/
typedef int IDXL_Comm_t;

/** Comm_begin begins a message exchange.  It is currently 
 * a collective routine, and exactly one exchange can be outstanding;
 * but these restrictions may be relaxed later. 
 * @param tag a user-defined "tag" for this exchange.
 * @param context an MPI communicator, or 0 for the default. 
 */
IDXL_Comm_t IDXL_Comm_begin(int tag, int context); 

/** Remote-copy this data on flush/wait. If m is zero, includes begin&wait. */
void IDXL_Comm_sendrecv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t l, void *data);
/** Remote-sum this data on flush/wait. If m is zero, includes begin&wait. */
void IDXL_Comm_sendsum(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t l, void *data);

/** Send this data out when flush is called. Must be paired with a recv or sum call */
void IDXL_Comm_send(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t l, const void *srcData);
/** Copy this data from the remote values when wait is called. */
void IDXL_Comm_recv(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t l, void *destData);
/** Add this data with the remote values when wait is called. */
void IDXL_Comm_sum(IDXL_Comm_t m,IDXL_t dest, IDXL_Layout_t l, void *sumData);

/** Send all outgoing data. */
void IDXL_Comm_flush(IDXL_Comm_t m);

/** Block until all communication is complete. This destroys the IDXL_Comm. */
void IDXL_Comm_wait(IDXL_Comm_t m);


/* Collective Operations */
#define IDXL_FIRST_REDTYPE 1520000000 /*first valid IDXL reduction type*/
#define IDXL_SUM (IDXL_FIRST_REDTYPE+0)
#define IDXL_PROD (IDXL_FIRST_REDTYPE+1)
#define IDXL_MAX (IDXL_FIRST_REDTYPE+2)
#define IDXL_MIN (IDXL_FIRST_REDTYPE+3)


#ifdef __cplusplus
  }
#endif

#endif
