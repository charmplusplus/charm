#define CblockingInvoke(BOC1, ep1 , m, g, p) CharmBlockingCall(GetEntryPtr(BOC1,ep1), m, g,p)
extern "C" SetRefNumber(void *, int);
extern "C" GetRefNumber(void *);

extern "C" void InitCharmFutures();
extern "C" void*  CharmBlockingCall(int entry, void * m, int g, int p);
extern "C" CthThread CthCreate(void (*)(...), void *, int);

extern "C" void CSendToFuture(void *m, int processor);
