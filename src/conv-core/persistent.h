
typedef int PersistentHandle;

#if CMK_PERSISTENT_COMM

void CmiPersistentInit();
PersistentHandle CmiCreatePersistent(int destPE, int maxBytes);
void CmiUsePersistentHandle(PersistentHandle *p, int n);
void CmiDestoryAllPersistent();

#else

#define CmiPersistentInit()
#define CmiCreatePersistent(x,y)  0
#define CmiUsePersistentHandle(x,y)
#define CmiDestoryAllPersistent()

#endif
