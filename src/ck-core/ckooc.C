#include "ckooc.h"
#include "oocAPI.h"
#include "pcqueue.h"
#include <pthread.h>
#include "OOC.decl.h"
#include <sys/mman.h>
/*readonly*/ long CkOOCMemLimit; 
/*readonly*/ std::string CkOOCPrefix;
/*readonly*/ int CkOOCMode;
/*readonly*/ CProxy_CkOOCManager CkOOCProxy;

CpvDeclare(CkOOCManager *, oocMgr);
CpvDeclare(std::vector<CkIOMetaData *>, oocParams);
long OOCInMemSize;
char * OOCBuffer;
pthread_mutex_t IOLock;
pthread_cond_t IOSig;
int readReqs;
int writeReqs;

class prioComparison
{
  public:
  bool operator() (OOCTask * & lhs, OOCTask * & rhs) const
  {
    return (lhs->prio>rhs->prio);
  }
};

#if USE_PRIO_Q
#include <queue>
typedef std::priority_queue<OOCTask *,std::deque<OOCTask *>, prioComparison> PrioIOQ;
PrioIOQ pendingTasks;
#elif USE_FIFO_Q
std::list<OOCTask *> pendingTasks;
#else
std::map<int, std::list<OOCTask *> > pendingTasks;
#endif
//list of available data units: not used by other ready computation tasks
std::list<int> availList;//no update since last read
std::list<int> availDirtyList; //requires a real write
//list of all the metadata of CkIOHandles
std::vector<CkIOMetaData *> IOHandleList;
//whether IO thread is actively scheduling tasks from pendingTasks
bool IOSchedulerActive;

void createThread(int core);
extern void initNodeQueue();

void exitOOCStart(){
  CkOOCProxy[0].printAnalysis();
}

extern "C" void set_thread_affinity(int cpuid);

void initOOC(char ** argv)
{
  CpvInitialize(CkOOCManager *, oocMgr);
  CpvInitialize(std::vector<CkIOMetaData *>, oocParams);
  if(CkMyRank() == 0){
    IOSchedulerActive = false;
    OOCInMemSize = 0;
    initNodeQueue();
    pthread_mutex_init(&IOLock,NULL);
    pthread_cond_init(&IOSig,NULL);
    
    //find the core that comm thread is mapped to
    int core;
    char *commap = NULL;
    CmiGetArgStringDesc(argv, "+commap", &commap, "define comm threads to core mapping");
    if(commap == NULL){
      CkPrintf("comm thread mapping (used for IO thread as well in out-of-core mode) is not set, out-of-core is disabled\n");
      return;
    }

    char *mapstr = (char*)malloc(strlen(commap)+1);
    strcpy(mapstr, commap);              
    char * ptr = NULL;          
    char * str = strtok_r(mapstr, ",", &ptr);
    sscanf(str, "%d", &core);
 
    createThread(core);
    registerExitFn(exitOOCStart);
  }
}

void * executeIOTasks(void * id){
  set_thread_affinity(*(int *)id);

  readReqs = 0;
  writeReqs = 0;
  //set affinity
  //pthread_t thread = pthread_self();
  //cpu_set_t cpuset;
  //CPU_ZERO(&cpuset);
  //TODO map to the core of communication thread
  //CPU_SET(CmiMyNodeSize()+1, &cpuset);
  //int status = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  //if(status != 0)
  //  CkAbort("Fail to pin the IO thread");
  
  bool hasTask = false;
  std::list<std::pair<CkIOMetaData *, CkIOMetaData *> > IOList;
  int reqSize;
  OOCTask * candidate;
  while(1){
    pthread_mutex_lock(&IOLock);
    if(!hasTask){
      IOSchedulerActive = false;
      pthread_cond_wait(&IOSig, &IOLock);
    }

    hasTask = false;
#if !USE_FIFO_Q&&!USE_PRIO_Q    
    //availability queue
    for(auto it = pendingTasks.begin(); it != pendingTasks.end(); ++it){
      reqSize = it->first;
      if(it->second.size() == 0)
        continue;
      candidate = it->second.front();
#elif USE_FIFO_Q
    //fifo queue  
    if(!pendingTasks.empty()){
      candidate = pendingTasks.front();
      reqSize = candidate->bringInList.size();
#else
    //priority queue  
    candidate = NULL;
    if(!pendingTasks.empty()) {
      candidate = pendingTasks.top();
      //some tasks may already have been processed when fetching data for other tasks
      while(candidate == NULL || candidate->dead){
        candidate->deleteCount++;
        if(candidate->deleteCount == 2)
          delete candidate;
        pendingTasks.pop();
        if(pendingTasks.empty())
          break;
        candidate = pendingTasks.top();
      }
    }
    if(candidate != NULL && !candidate->dead){
      candidate->deleteCount++;
      reqSize = candidate->bringInList.size();
#endif
      //TODO: here assuming each data unit is of the same size
      int bufSize = IOHandleList[candidate->bringInList.front()]->getSize();
      int availNum = (CkOOCMemLimit-OOCInMemSize)/bufSize;

      availNum = availNum>reqSize?reqSize:availNum;
      reqSize = reqSize-availNum;
      std::vector<int> rmList;
      
      if(reqSize > 0){
        for(auto ij = availList.begin(); ij != availList.end(); ++ij){
          int idx = *ij;
          CkIOMetaData * md = IOHandleList[idx];
#if USE_DEP          
          if(!md->hasTask(candidate) && md->getTaskSize() == 0){
#else          
          if(!md->hasTask(candidate)){
#endif            
            rmList.push_back(idx);
            if(rmList.size() == reqSize){
              hasTask = true;
              break;
            }
          }
        }
#if USE_DEP        
        if(!hasTask){
          for(auto ij = availList.begin(); ij != availList.end(); ++ij){
            int idx = *ij;
            CkIOMetaData * md = IOHandleList[idx];
            if(!md->hasTask(candidate) && md->getTaskSize() != 0){
              rmList.push_back(idx);
              if(rmList.size() == reqSize){
                hasTask = true;
                break;
              }
            }
          }
        }
#endif        
#if USE_DIRTY        
        if(!hasTask){
          for(auto ij = availDirtyList.begin(); ij != availDirtyList.end(); ++ij){
            int idx = *ij;
            CkIOMetaData * md = IOHandleList[idx];
#if USE_DEP            
            if(!md->hasTask(candidate) && md->getTaskSize() == 0){
#else            
            if(!md->hasTask(candidate)){
#endif              
              rmList.push_back(idx);
              if(rmList.size() == reqSize){
                hasTask = true;
                break;
              }
            }
          }
        }
#if USE_DEP            
        if(!hasTask){
          for(auto ij = availDirtyList.begin(); ij != availDirtyList.end(); ++ij){
            int idx = *ij;
            CkIOMetaData * md = IOHandleList[idx];
            if(!md->hasTask(candidate) && md->getTaskSize() != 0){
              rmList.push_back(idx);
              if(rmList.size() == reqSize){
                hasTask = true;
                break;
              }
            }
          }
        }
#endif        
#endif        
      }
      else
        hasTask = true;
      
      //succeed
      if(hasTask){
        candidate->incRefCount();
        IOSchedulerActive = true;
#if USE_PRIO_Q
        pendingTasks.pop();
#elif USE_FIFO_Q        
        pendingTasks.pop_front();
#else
        it->second.pop_front();
#endif
        for(auto ij = rmList.begin(); ij != rmList.end(); ++ij){
          int idx = *ij;
          CkIOMetaData * md = IOHandleList[idx];
          availList.remove(idx);
          availDirtyList.remove(idx);
          md->addToBringList();
          md->setState(INDISK);
          md->reset();
        }
        for(int i = 0 ; i < availNum; i++){
          int bringInIdx = candidate->bringInList.front();
          candidate->bringInList.pop_front();
          CkIOMetaData * rdmd = IOHandleList[bringInIdx];
          rdmd->setBuf(OOCBuffer+OOCInMemSize);
          OOCInMemSize+=rdmd->getSize();
          rdmd->removeFromBringList(candidate);
          rdmd->setState(TOMEM);
          CkIOMetaData * wrmd = NULL;
          IOList.push_back(std::make_pair(wrmd, rdmd));
        }
        for(auto ij = rmList.begin(); ij != rmList.end(); ++ij){
          int idx = *ij;
          CkIOMetaData * wrmd = IOHandleList[idx];
          int bringInIdx = candidate->bringInList.front();
          candidate->bringInList.pop_front();
          //printf("take out %d for %d\n", idx, bringInIdx);
          CkIOMetaData * rdmd = IOHandleList[bringInIdx];
          rdmd->removeFromBringList(candidate);
          rdmd->setState(TOMEM);
          IOList.push_back(std::make_pair(wrmd, rdmd));
        }
      }
#if !USE_FIFO_Q&&!USE_PRIO_Q     
      break;
#endif    
    }
    pthread_mutex_unlock(&IOLock);
    
    //perform write read
    if(hasTask){
      int size;
      int fd;
      while(IOList.size() != 0){
        std::pair<CkIOMetaData *, CkIOMetaData *> & p = IOList.front();
        CkIOMetaData * wrmd = p.first;
        CkIOMetaData * rdmd = p.second;
        
        //need to make space
        if(wrmd!=NULL){
          //write operations
          if(wrmd->isRealWrite()){
            fd = open(wrmd->getFile(), O_WRONLY|O_DIRECT);
            if(fd<=0)
              assert(false);
            size = write(fd, wrmd->getBuf(), wrmd->getSize());
            if(size != wrmd->getSize()){
              printf("write error: write size %d actural size %d\n", size, wrmd->getSize());
              assert(false);
            }
            fsync(fd);
            close(fd);
            writeReqs++;
          }
          rdmd->setBuf(wrmd->getBuf());
          wrmd->setBuf(NULL);
        }
        
        //read operations
        if(rdmd->getType() != WRONLY && rdmd->getType() != BUFFER){
          fd = open(rdmd->getFile(), O_RDONLY|O_DIRECT);
          if(fd<=0)
            assert(false);
         size = read(fd, rdmd->getBuf(), rdmd->getSize());
          close(fd);
          if(size != rdmd->getSize()){
            printf("read error: write size %d actural size %d\n", size, rdmd->getSize());
            assert(false);
          }
          readReqs++;
        }
        
        rdmd->setState(INMEM);
        pthread_mutex_lock(&IOLock);
        rdmd->checkReady();
        pthread_mutex_unlock(&IOLock);
        IOList.pop_front();
      }
    }
  }
}

void createThread(int core){
  pthread_t thread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&thread,&attr,executeIOTasks, &core);
}

void scheduleTask(OOCTask * task){
  CpvAccess(oocMgr)->scheduleTask(task);
}

int addHandler(CkIOMetaData * md){
  return CpvAccess(oocMgr)->addHandler(md);
}

CkOOCManager::CkOOCManager(){
  dirCreated = false;
  CpvAccess(oocMgr) = thisProxy.ckLocalBranch();
}

int CkOOCManager::addHandler(CkIOMetaData * md){
  if(!dirCreated){
    char name[64];
    sprintf(name,"%s/%d", CkOOCPrefix.c_str(), CkMyPe());
    CkPrintf("[%d]create directory %s\n",CkMyPe(), name);
    CmiMkdir(name);
    dirCreated = true;
  }
  int size = md->getSize();
  bool inMem = false;
  int idx;
  long offset;
  pthread_mutex_lock(&IOLock);
  IOHandleList.push_back(md);
  idx = IOHandleList.size()-1;
  if(OOCInMemSize+size<=CkOOCMemLimit){
    inMem = true;
    availList.push_back(idx);
    offset = OOCInMemSize;
    OOCInMemSize+=size;
  }
  pthread_mutex_unlock(&IOLock);
  
  if(inMem){
    char * buf = (char *)(OOCBuffer+offset);
    md->setBuf(buf);
    md->setState(INMEM);
  }
  
  return idx;
}

void CkOOCManager::scheduleTask(OOCTask * task){
  pthread_mutex_lock(&IOLock);
  if(task->checkReady()){
    delete task;
  }
  else{
    task->addToList();
    int pendingSize = task->bringInList.size();
    if(pendingSize>0){
#if USE_PRIO_Q
      //priority queue
      pendingTasks.push(task);
#elif USE_FIFO_Q      
      //fifo queue
      pendingTasks.push_back(task);
#else      
      //based on data availability
      pendingTasks[pendingSize].push_back(task);
#endif      
    //signal the IOThread if idle
      if(!IOSchedulerActive){
        pthread_cond_signal(&IOSig);
      }
    }else{
      task->incRefCount();
    }
  }
  pthread_mutex_unlock(&IOLock);
}

void CkOOCManager::printAnalysis(){
  CkPrintf("[%d]IO request count: read %d write %d\n", CkMyNode(), readReqs, writeReqs);
  CkExit();
}

//TODO easy way tp specify size
void OOCInit(long size, std::string prefix, int mode){
  CkOOCMemLimit = size*1048576;
  CkOOCPrefix = prefix;
  CkOOCMode = mode;
  CkOOCProxy = CProxy_CkOOCManager::ckNew();
  CkPrintf("OOC set memory limit %ld\n", CkOOCMemLimit);
  
  int err = posix_memalign((void**)&(OOCBuffer),4096,CkOOCMemLimit);
  if(err != 0)
    assert(false);
}

//LOCK PROTECTED!!
void CkIOMetaData::addToBringList(){
  for(auto ik = taskList.begin(); ik != taskList.end(); ++ik){  
    (*ik)->bringInList.push_back(idx);
#if !USE_FIFO_Q&&!USE_PRIO_Q    
    int newSize = (*ik)->bringInList.size();
    pendingTasks[newSize-1].remove(*ik);
    pendingTasks[newSize].push_back(*ik);
#endif    
  }
}

//LOCK PROTECTED!!
void CkIOMetaData::removeFromBringList(OOCTask * candidate){
  for(auto ik = taskList.begin(); ik != taskList.end(); ++ik){
    if(*ik != candidate){
      (*ik)->bringInList.remove(idx);
      int newSize = (*ik)->bringInList.size();
#if USE_PRIO_Q
      if(newSize == 0){
        (*ik)->incRefCount();
      }
#elif USE_FIFO_Q
      if(newSize == 0){
        pendingTasks.remove(*ik);
        (*ik)->incRefCount();
      }
#else
      pendingTasks[newSize+1].remove(*ik);
      if(newSize > 0)
        pendingTasks[newSize].push_back(*ik);
      else
        (*ik)->incRefCount();
#endif        
    }
  }
}

//LOCK PROTECTED!!
void CkIOMetaData::checkReady(){
  for(auto it = taskList.begin(); it != taskList.end();){
    OOCTask * t = *it;
    if(t->checkReady()){
      for(auto ij = t->dependences.begin(); ij != t->dependences.end(); ++ij){
        if(*ij != this){
          (*ij)->removeTask(t);
        }
      }
      it = taskList.erase(it);
#if USE_PRIO_Q      
      t->deleteCount++;
      if(t->deleteCount == 2)
#endif
        delete t;
    }
    else
      it++;
  }
}

void OOCWriteAll(){
  for(auto it = availDirtyList.begin(); it != availDirtyList.end(); ++it){
    int idx = *it;
    CkIOMetaData * md = IOHandleList[idx];
    md->setState(INDISK);
    md->reset();
    int fd = open(md->getFile(), O_WRONLY|O_DIRECT);
    if(fd<=0)
      assert(false);
    int size = write(fd, md->getBuf(), md->getSize());
    assert(size == md->getSize());
    fsync(fd);
    close(fd);
  }
  
  for(auto it = availList.begin(); it != availList.end(); ++it){
    int idx = *it;
    CkIOMetaData * md = IOHandleList[idx];
    md->setState(INDISK);
    md->reset();
    int fd = open(md->getFile(), O_WRONLY|O_DIRECT);
    if(fd<=0)
      assert(false);
    int size = write(fd, md->getBuf(), md->getSize());
    assert(size == md->getSize());
    fsync(fd);
    close(fd);
  }
  
  availList.clear();
  availDirtyList.clear();
  OOCInMemSize = 0;
}

#include "OOC.def.h"
