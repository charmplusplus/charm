#include <converse.h>

typedef int (*CmsWorkerFn) (void *, void *);
typedef int (*CmsConsumerFn) (void *, int);

void CmsInit(CmsWorkerFn f, int maxResponses);
void CmsFireTask(int ref, void *t, int size);
void CmsAwaitResponses(void);
void CmsProcessResponses(CmsConsumerFn f);
void *CmsGetResponse (int ref);
void CmsExit(void);
