/******************************************************************************
 * File: MPI_Win.C
 *       This file implenments routines of one-sided communication.
 * Author: Yan Shi (yanshi@uiuc.edu)
 * Last Revision: 2003/10/30 
 ******************************************************************************/
#include "ampiimpl.h"

/*
 * int MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
 *	       MPI_Info info, MPI_Comm comm, MPI_Win *newwin)
 *   Creates the window object and returns the pointer for *win
 *
 *   ---Assumption: memory location at *base is pre-allocated 
 *   ---by a MPI_Alloc_mem call
 *
 *   Inputs:
 *     void *base : pointer specifying the memory area to create the window
 *     ????? MPI_Aint size : size of target memory area (in bytes)
 *     ????? int disp_unit : number of bytes for one datatype
 *     MPI_Info info : MPI_Info object, provides hints for optimization
 *     MPI_Comm comm : communicator 
 *     MPI_Win *newwin : stores the handle to the created MPI_Win object on return
 *  
 *   Returns int: MPI_SUCCESS or MPI_ERR_WIN
 */
// A collective call over all processes in the communicator
// MPI_Win object created LOCALLY on all processes when the call returns
CDECL
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit,
	       MPI_Info info, MPI_Comm comm, MPI_Win *newwin){
  AMPIAPI("MPI_Win_create");
  ampi *ptr = getAmpiInstance(comm);
 
  *newwin = ptr->createWinInstance(base, size, disp_unit, info);

  /* need to reduction here: to make sure every processor participates */
  MPI_Barrier(comm);

  if (newwin)
    return MPI_SUCCESS;
  else
    return MPI_ERR_WIN;
}

/*
 * int MPI_Win_free(MPI_Win *win):
 *   Frees the window object and returns a null pointer for *win
 */
// A collective call over all processes in the communicator
// MPI_Win object deleted LOCALLY on all processes when the call returns
CDECL
int MPI_Win_free(MPI_Win *win){
  AMPIAPI("MPI_Win_free");
  if(win==NULL) {
    return MPI_ERR_WIN;
  }

  ampi *ptr = getAmpiInstance(win->comm);
  win_obj *winobj = ptr->getWinObjInstance(*win);
  MPI_Comm comm = win->comm;

  /* Need a barrier here: to ensure that every process participates */
  MPI_Barrier(comm);

  ptr->deleteWinInstance(winobj);
  win = NULL;

  return MPI_SUCCESS;
}

CDECL
int MPI_Win_delete_attr(MPI_Win win, int key){
  AMPIAPI("MPI_Win_delete_attr");
  return MPI_SUCCESS;
}

/*
 * ---Note : No atomicity for overlapping Puts. 
 * ---sync calls should be made on this window to ensure the 
 * ---correctness of the operation
 */
CDECL
int MPI_Put(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win){
  AMPIAPI("MPI_Put");
  ampi *ptr = getAmpiInstance(win.comm);
  return ptr->winPut(orgaddr, orgcnt, orgtype, rank, targdisp, targcnt, targtype, win);
}

/*
 * ---Note : No atomicity for overlapping Gets. 
 * ---sync calls should be made on this window to ensure the 
 * ---correctness of the operation
 */
CDECL
int MPI_Get(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank, 
	MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, MPI_Win win){
  AMPIAPI("MPI_Get");
  ampi *ptr = getAmpiInstance(win.comm);
  
  // winGet is a local function which will call the remote method on #rank processor 
  return  ptr->winGet(orgaddr, orgcnt, orgtype, rank, targdisp, targcnt, targtype, win);
}
	

/*
 * int MPI_Accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
 *		   MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
 *		   MPI_Op op, MPI_Win win)
 *   Accumulates the contents from the origin buffer to the target area using
 *   the predefined op operation.
 *
 * ---Accumulate call is ATOMIC: no sync is needed 
 * ---Many accumulate can be made from many origins to one target
 * ---??? Is this ensured by charm automatically???
 *
 */
CDECL
int MPI_Accumulate(void *orgaddr, int orgcnt, MPI_Datatype orgtype, int rank,
		   MPI_Aint targdisp, int targcnt, MPI_Datatype targtype, 
		   MPI_Op op, MPI_Win win) {
  AMPIAPI("MPI_Accumulate");
  ampi *ptr = getAmpiInstance(win.comm);  		   
  return  ptr->winAccumulate(orgaddr, orgcnt, orgtype, rank, 
                         targdisp, targcnt, targtype, op, win);   
}


/*
 * int MPI_Win_fence(int assertion, MPI_Win win)
 *   Synchronizes all one-sided communication calls on this MPI_Win.   
 *   (Synchronized RMA operations on the specified window)
 *
 *   Inputs: 
 *     int assertion : program assertion, used to provide optimization hints
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int MPI_Win_fence(int assertion, MPI_Win win){
  AMPIAPI("MPI_Win_fence");
  MPI_Comm comm = win.comm;
  ampi *ptr = getAmpiInstance(comm);

  // Wait until everyone reaches the fence
  MPI_Barrier(comm);

  // Complete all outstanding one-sided comm requests
  // no need to do this for the pseudo-implementation
  return MPI_SUCCESS;
}


/*
 * int MPI_Win_lock(int lock_type, int rank, int assertion, MPI_Win win)
 *   Locks access to this MPI_Win object.   
 *   Input:
 *     int lock_type : MPI_LOCK_EXCLUSIVE or MPI_LOCK_SHARED
 *     int rank : rank of locked window
 *     int assertion : program assertion, used to provide optimization hints
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int MPI_Win_lock(int lock_type, int rank, int assertion, MPI_Win win){
  AMPIAPI("MPI_Win_lock");
  ampi *ptr = getAmpiInstance(win.comm);

  // process assertion here: HOW???  
  // end of assertion
  ptr->winLock(lock_type, rank, win);
  return MPI_SUCCESS;
}



/*
 * int MPI_Win_unlock(int rank, MPI_Win win)
 *   Unlocks access to this MPI_Win object.   
 *   Input:
 *     int rank : rank of locked window
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
// The RMA call is completed both locally and remotely after unlock. 
CDECL
int MPI_Win_unlock(int rank, MPI_Win win){
  AMPIAPI("MPI_Win_unlock");
  ampi *ptr = getAmpiInstance(win.comm);

  // process assertion here: HOW???  
  // end of assertion
  ptr->winUnlock(rank, win);
  return MPI_SUCCESS;
}

/* the following four functions are yet to implement */
/*
 * int MPI_Win_post(MPI_Group group, int assertion, MPI_Win win)
 *   Opens a RMA access epoch for local window win.
 *   Only processes in group can access this window with RMA calls.
 *   Each process must issue a matching MPI_Win_start to start the 
 *     access epoch.
 *   Post is non-blocking while start could be blocking.
 *   Input:
 *     MPI_Group group : a group of processes 
 *   Returns int : MPI_SUCCESS or MPI_ERR_WIN
 */
CDECL
int MPI_Win_post(MPI_Group group, int assertion, MPI_Win win){
  AMPIAPI("MPI_Win_post");
  ampi *ptr = getAmpiInstance(win.comm);
  //  ptr->winPost(group, win);
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_wait(MPI_Win win){
  AMPIAPI("MPI_Win_wait");
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_start(MPI_Group group, int assertion, MPI_Win win){
  AMPIAPI("MPI_Win_start");
  ampi *ptr = getAmpiInstance(win.comm);
  //  ptr->winStart(group, win);
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_complete(MPI_Win win){
  AMPIAPI("MPI_Win_complete");
  return MPI_SUCCESS;
}


/*
 * int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr) 
 *   A simple wrapper around 'malloc' call. Used to allocate memory
 *   for MPI functions
 *
 *   Inputs: 
 *     MPI_Aint size : size of target memory area
 *     MPI_Info info : 
 *     void *base : stores the base pointer of target memory area on return
 *   Return: 
 *     void* : address of the allocated memory
 */
CDECL
void* MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr){
  AMPIAPI("MPI_Alloc_mem");
  baseptr = malloc(size);
  return baseptr;
}


/*
 * int MPI_Free_mem(void *base)
 *   Frees memory that was previous allocated by MPI_Alloc_mem call
 */
CDECL
int MPI_Free_mem(void *baseptr){
  AMPIAPI("MPI_Free_mem");
  free(baseptr);
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_get_group(MPI_Win win, MPI_Group *group) {
  AMPIAPI("MPI_Win_get_group");
  ampi *ptr = getAmpiInstance(win.comm);
  ptr->winGetGroup(win, group);
  return MPI_SUCCESS;

}

CDECL
int MPI_Win_set_attr(MPI_Win win, int key, void* value) {
  AMPIAPI("MPI_Win_set_attr");
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_set_name(MPI_Win win, char *name) {
  AMPIAPI("MPI_Win_set_name");
  ampi *ptr = getAmpiInstance(win.comm);
  ptr->winSetName(win, name);
  return MPI_SUCCESS;
}

CDECL
int MPI_Win_get_name(MPI_Win win, char *name, int *length) {
  AMPIAPI("MPI_Win_get_name");
  ampi *ptr = getAmpiInstance(win.comm);
  ptr->winGetName(win, name, length);
  return MPI_SUCCESS;
}

