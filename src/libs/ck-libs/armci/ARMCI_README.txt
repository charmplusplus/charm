Download nightly build from last night:
http://charm.cs.uiuc.edu/autobuild/bin/charm_src.tar.gz

Build charm target ARMCI (instead of charm or AMPI)
> cd charm
> ./build ARMCI net-linux -O3
Compiling the code with:
> charm/bin/charmc -c lu.c $(OPTS)
Linking the program with:
> charm/bin/charmc lu.o -o lu.pgm -swapglobals -memory isomalloc -language armci $(OPTS)

Run the program
> ./charmrun ./pgm +p2 +vp8 

main function has to be compliant to ANSI C:
int main(int argc, char *argv[]);

Following is a list of implemented (and yet-to-implement) functions.

/* Functions already implemented */
int ARMCI_Procs(int *procs);
int ARMCI_Myid(int *myid);

void ARMCI_Migrate(void);
void ARMCI_Async_Migrate(void);
void ARMCI_Checkpoint(char* dirname);
void ARMCI_MemCheckpoint(void);

int ARMCI_Init(void);
int ARMCI_Finalize(void);
void ARMCI_Error(char *msg, int code);
void ARMCI_Cleanup(void);

int ARMCI_Put(...);
int ARMCI_NbPut(...);
int ARMCI_Get(...);
int ARMCI_NbGet(...);
  
int ARMCI_PutS(...);
int ARMCI_NbPutS(...);
int ARMCI_GetS(...);
int ARMCI_NbGetS(...);

int ARMCI_Wait(armci_hdl_t *handle);
int ARMCI_WaitProc(int proc);
int ARMCI_WaitAll();
int ARMCI_Test(armci_hdl_t *handle);
int ARMCI_Barrier();

int ARMCI_Fence(int proc);
int ARMCI_AllFence(void);

int ARMCI_Malloc(void* ptr_arr[], int bytes);
int ARMCI_Free(void *ptr);
void *ARMCI_Malloc_local(int bytes);
int ARMCI_Free_local(void *ptr);
 
int armci_notify(int proc);
int armci_notify_wait(int proc, int *pval);

/* Functions yet to implement */
int ARMCI_GetV(...);
int ARMCI_NbGetV(...);
int ARMCI_PutV(...);
int ARMCI_NbPutV(...);
int ARMCI_AccV(...);
int ARMCI_NbAccV(...);

int ARMCI_Acc(...);
int ARMCI_NbAcc(...);
int ARMCI_AccS(...);
int ARMCI_NbAccS(...);

int ARMCI_PutValueLong(long src, void* dst, int proc);
int ARMCI_PutValueInt(int src, void* dst, int proc);
int ARMCI_PutValueFloat(float src, void* dst, int proc);
int ARMCI_PutValueDouble(double src, void* dst, int proc);
int ARMCI_NbPutValueLong(long src, void* dst, int proc, armci_hdl_t* handle);
int ARMCI_NbPutValueInt(int src, void* dst, int proc, armci_hdl_t* handle);
int ARMCI_NbPutValueFloat(float src, void* dst, int proc, armci_hdl_t* handle);
int ARMCI_NbPutValueDouble(double src, void* dst, int proc, armci_hdl_t* handle);
long ARMCI_GetValueLong(void *src, int proc);
int ARMCI_GetValueInt(void *src, int proc);
float ARMCI_GetValueFloat(void *src, int proc);
double ARMCI_GetValueDouble(void *src, int proc);

void ARMCI_SET_AGGREGATE_HANDLE (armci_hdl_t* handle);
void ARMCI_UNSET_AGGREGATE_HANDLE (armci_hdl_t* handle);

int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc);
int ARMCI_Create_mutexes(int num);
int ARMCI_Destroy_mutexes(void);
void ARMCI_Lock(int mutex, int proc);
void ARMCI_Unlock(int mutex, int proc);

