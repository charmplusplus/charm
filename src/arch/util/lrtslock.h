#ifndef _LRTS_LOCK_DECL_H
#define _LRTS_LOCK_DECL_H

#if CMK_SHARED_VARS_UNAVAILABLE
typedef int LrtsNodeLock;
#else
#if CMK_SHARED_VARS_NT_THREADS /*Used only by win versions*/
typedef HANDLE LrtsNodeLock;
#else
typedef void* LrtsNodeLock;
#endif
#endif //CMK_SHARED_VARS_UNAVAILABLE

LrtsNodeLock LrtsCreateLock(void);
void LrtsLock(LrtsNodeLock lock);
void LrtsUnlock(LrtsNodeLock lock);
int LrtsTryLock(LrtsNodeLock lock);
void LrtsDestroyLock(LrtsNodeLock lock);

#define CmiNodeLock LrtsNodeLock
#define CmiCreateLock LrtsCreateLock
#define CmiLock(l) LrtsLock((LrtsNodeLock)l)
#define CmiUnlock(l) LrtsUnlock((LrtsNodeLock)l)
#define CmiTryLock(l) LrtsTryLock((LrtsNodeLock)l)
#define CmiDestroyLock(l) LrtsDestroyLock((LrtsNodeLock)l)


#endif //_LRTS_LOCK_DECL_H
