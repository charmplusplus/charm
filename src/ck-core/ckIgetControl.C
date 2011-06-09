#include "charm++.h"
#include "ckIgetControl.h"
#ifndef WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif
#include <unistd.h>
#include <sys/types.h>

IGetControlClass TheIGetControlClass;
int IGET_TOTALMEMORY = 480000000;

#if IGET_FLOWCONTROL==0
#elif CKFUTURE_IGET

int getAvailMemory(int grainsize);

int 
IGetControlClass::IGetControlClass::iget_request(CkIGetID fut, void *msg,
int ep, CkArrayID id, CkArrayIndex idx,
void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int),int msgsize)
 {
    if(msgsize>0)  IGET_UNITMESSAGE=msgsize;
    int ret_status=1, size=1;
    if(iget_token>=size){
      iget_token-=size;
      iget_outstanding++;
      //(fptr)(obj,msg,ep,0);  // Send the msg here
    }
    else //(iget_request(CthSelf(),1)==false)
    {
      iget_tokenqueue_enqueue(fut,msg,ep,id,idx,fptr);
      ret_status = 0; // No send will be done this case
    }
    return ret_status;
  }

void 
IGetControlClass::iget_free(int size)
  {
    iget_token+=size;
    iget_outstanding--;
    // if some one not sent yet
    iget_tokenqueue_entry e=iget_tokenqueue_dequeue();
    if(e!=NULL) 
    {
       iget_token-=size;
       iget_outstanding++;
    //   (e->fptr)(e->obj,e->m, e->ep, 0);
	(e->fptr)(e->aid, e->idx, e->m, e->ep, 0);
       delete e;
     }
  }

void
IGetControlClass::iget_updateTokenNum() {
    double currenttime = CmiWallTimer();
    if(currenttime-lastupdatetime<1)
       return;
    int totalMemUsed = (int)getRSS();
    if(totalMemUsed<=0) return;
    int leftMem = IGET_TOTALMEMORY-totalMemUsed;
    int iget_token_new = (leftMem)/(int)IGET_UNITMESSAGE;
    if(leftMem<0 || iget_token_new<IGET_MINTOKENNUM) {lastupdatetime =
currenttime;return;} //iget_token_new = IGET_MINTOKENNUM;
    //CmiResetMaxMemory();
    lastupdatetime = currenttime;
    iget_token_history = iget_token ;
    iget_token = iget_token_new - iget_outstanding;
    if(iget_token<0) iget_token=0;
//  iget_token = 1/3(iget_token_new + iget_token + iget_token_history);
    printf("availMem %d,  IGET_UNIT %d, out_standinging %d\n",
     (leftMem), IGET_UNITMESSAGE, iget_outstanding);
   printf("Update Token num from %d to %d\n", iget_token_history, iget_token);
  }

/*
void 
IGetControlClass::iget_updateTokenNum() {
    double currenttime = CmiWallTimer();
    if(currenttime-lastupdatetime<1)
       return; 
//    int totalMemUsed = (int)CmiMaxMemoryUsage();
    int leftMem = (int)getAvailMemory((int)IGET_UNITMESSAGE); //IGET_TOTALMEMORY-totalMemUsed;
    int iget_token_new = (leftMem)/(int)IGET_UNITMESSAGE;
    if(leftMem<0 || iget_token_new<IGET_MINTOKENNUM) {lastupdatetime =
currenttime;return;} //iget_token_new = IGET_MINTOKENNUM;
    //CmiResetMaxMemory();
    lastupdatetime = currenttime;
    iget_token_history = iget_token ;
    iget_token = iget_token_new - iget_outstanding;
    if(iget_token<0) iget_token=0;
//  iget_token = 1/3(iget_token_new + iget_token + iget_token_history); 
    printf("availMem %d,  IGET_UNIT %d, out_standinging %d\n",
     (leftMem), IGET_UNITMESSAGE, iget_outstanding);
   printf("Update Token num from %d to %d\n", iget_token_history, iget_token);
  }
*/


/*
 *  Called when wait is posted, but no iget is really sent yet
 *  
 */
void 
IGetControlClass::iget_resend(CkIGetID fut)
  {
    // if found in the wait queue
    // else return and do nothing
//    if(!iget_tokenqueue_find(fut))
//      return;

    //promote self to head of wait queue
    //return, and sleep on wait for future
    //iget_tokenqueue_promote(fut);
    // NEED: to set status of this entry to 1 --> this has been waited for
  }



/* Get available memory in system accessible to the application 
 *   This is done by a while loop of allocating certain size of memory
 *   and keep those memory in core by pinning them page-by-page. The allocation
 *   size is determined by 'grainsize'
 */
int 
getAvailMemory(int grainsize) 
{

  struct rusage ru;
  double a = 1;
  int chunk = 50*1024*1024;
  int subchunk = 20*1024*1024;
  unsigned long size = chunk;
  char *data = NULL, *olddata = NULL;
  unsigned long init_pf = 0;
  int pagesize = getpagesize();
/*
  if(grainsize >= 10*1024*1024)
    subchunk = 20*1024*1024;    
  else if(grainsize >= 1024*1024)
    subchunk = 2*1024*1024;
  else
    subchunk = 256*1024;
*/
  while (1) {
    olddata = data;
    data = (char *)realloc(data, size);
    if (data == NULL) break;
    for (int i=size-chunk; i<size; i+=pagesize)  data[i]=i;
    for (int i=0; i<size; i+=pagesize)  a += data[i];
    getrusage(RUSAGE_SELF, &ru);
    if (size == chunk)                   // ignore init page faults
      init_pf = ru.ru_majflt;
    else
      if (ru.ru_majflt > init_pf) break;
    size += chunk;
  }
/*
  if(subchunk < chunk) {
    size += subchunk;
    while(1){
      olddata = data;
      data = (char *)realloc(data, size);
      if (data == NULL) break;
      for (int i=size-subchunk; i<size; i++)  data[i]=i;
      for (int i=0; i<size; i+=pagesize)  a *= data[i];
      getrusage(RUSAGE_SELF, &ru);
      if (size == chunk)                   // ignore init page faults
        init_pf = ru.ru_majflt;
      else
        if (ru.ru_majflt > init_pf) break;
      size += subchunk;
    }
  }
*/ 
  free(olddata); 
  return size;
}

extern "C"
void getAvailSysMem() {
  IGET_TOTALMEMORY = getAvailMemory(0); 
  printf("total physical memory : %d\n", IGET_TOTALMEMORY);
}

extern "C"
void TokenUpdatePeriodic()
{
  TheIGetControlClass.iget_updateTokenNum();
}

extern "C"
int getRSS()
{
  int ret=-1, i=0;
  pid_t pid;
  char filename[128], commands[256], retstring[128];
  int fd;
  pid = getpid();
  sprintf(filename,"__topmem__%d", CkMyPe());
  sprintf(commands, "export TERM=vt100; top -b -n 1 -p %d |grep %d | awk  -F' ' '{print $6}' > %s", pid,
pid, filename);
  system(commands);
  i=0;
  while(i<10){
    i++;
    fd = open(filename, O_RDONLY);
    if(fd>=0) break;
    else 
      printf("fileopen %s fails, try again\n", filename); 
  }
  if(fd<0){
    printf("fileopen %s fails, abort\n", filename); return -1;
  }
  lseek(fd, 0, SEEK_SET);
  i=0;
  while(read(fd, &retstring[i], sizeof(char))>0) i++;
  close(fd);
  ret = atoi(retstring);
  if(i>2){
    if(retstring[i-2]=='m') ret *= 1024*1024;
    if(retstring[i-2]=='k') ret *= 1024;
  }
//  sprintf(commands, "rm -f %s", filename);
//  system(commands);
  printf("RSS %d\n",ret);
  return ret;
}

#endif

