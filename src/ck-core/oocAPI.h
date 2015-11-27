#ifndef _OOCAPI_H
#define _OOCAPI_H
#include <string>
#include <vector>
#include "OOC.decl.h"
#include <assert.h>
/*readonly*/ extern std::string CkOOCPrefix;
/*readonly*/ extern int CkOOCMode;
extern pthread_mutex_t IOLock;
extern pthread_cond_t IOSig;
extern bool IOSchedulerActive;
class OOCTask;
extern std::list<int> availList;
extern std::list<int> availDirtyList;

enum{RDONLY = 0, RDWR = 1, WRONLY = 2, BUFFER = 3};
enum{INMEM, INDISK, TOMEM};
enum{OOC_ADAPTIVE, OOC_STATIC};

//prioritize to swap the data unit has not be written since last read
#define USE_DIRTY 1
//prioritize to swap the data unit that is not used by any pending tasks
#define USE_DEP 1
//different queuing strategy, by default the queue based on availability is used
#define USE_FIFO_Q 0
#define USE_PRIO_Q 0

class CkIOMetaData
{
  template <typename T>
  friend class CkIOHandle;
private:  
  int size;
  int idx;
  int state; //INMEM, INDISK, TOMEM
  int type; //RDONLY, WRONLY, RDWR, BUFFER
  int refCount;
  std::list<OOCTask *> taskList;
  bool dirty;
  char * filename;
  int rank;
  bool realWrite;
  char * buf;
public:  
  void init()
  {
    dirty = false;
    realWrite = false;
    refCount = 0;
    state = INDISK;
    rank = CmiMyRank();
    taskList.resize(0);
  }
  
  int getSize(){return size;}
  
  void setBuf(char * _buf){
    buf = _buf;
  }
  char * getBuf(){
    return buf;
  }

  void setType(int _type){type = _type;}
  int getType(){return type;}

  void setState(int _state){state = _state;}
  int getState(){return state;}

  void setID(int _idx){
    idx = _idx;
    //create file for the CkIOHandle
    filename = new char[64];
    sprintf(filename,"%s/%d/%d", CkOOCPrefix.c_str(), CkMyPe(), idx);
    int fd = open(filename,O_CREAT|O_RDWR|O_DIRECT, S_IRUSR|S_IWUSR);
    if(fd<=0)
      assert(false);
    close(fd);
  }
  int getID(){return idx;}

  char * getFile(){return filename;}

  void decRefCount(){
    refCount--;
    assert(refCount>=0);
    if(type == WRONLY || type == RDWR){
      dirty = true;
    }
    else{
      dirty = false||dirty;
    }
    if(refCount == 0){
#if USE_DIRTY      
      if(dirty)
        availDirtyList.push_back(idx);
      else
#endif        
        availList.push_back(idx);
    }
  }
  void incRefCount(){
    refCount++;
    if(refCount == 1){             
      availList.remove(idx);       
      availDirtyList.remove(idx);  
    } 
  }

  void reset(){
    if(dirty)
      realWrite = true;
    else
      realWrite = false;
    dirty = false;  
  }
  bool isRealWrite(){
    return realWrite;
  }

  void addTask(OOCTask * task){
    taskList.push_back(task);
  }
  int getTaskSize(){
    return taskList.size();
  }
  void removeTask(OOCTask * task){
    taskList.remove(task);
  }
  bool hasTask(OOCTask * task){
    for(auto it = taskList.begin(); it != taskList.end(); ++it){
      if(*it == task)
        return true;
    }
    return false;
  }

  void addToBringList();
  void removeFromBringList(OOCTask * candidate);
  void checkReady();
};

int addHandler(CkIOMetaData * md);

template <typename T>
class CkIOHandle
{
  CkIOMetaData md;
  int size;
  public:
  CkIOHandle(){
    md.init();
  }
  
  //TODO: limitation: size can only be set once & all CkIOHandles have the same size
  void resize(int _size){
    size = _size;
    md.size = _size*sizeof(T);
    md.setID(addHandler(&md));
  }
  int getSize(){return size;}
  
  T & operator[] (std::size_t idx){
    return ((T *)(md.buf))[idx];
  }
  
  CkIOMetaData * getMD(){return &md;}
  T * getData(){return (T *)(md.getBuf());} 
  void setType(int _type){md.type = _type;}
};

extern std::queue<CkMigratable *> objQ;
extern CmiNodeLock objQLock;
extern std::vector<std::queue<CkTask *>> privateObjQ;

class OOCTask{
    int epIdx;
    CkMigratable * obj;
    char * msg;
    bool doFree;
    bool refCountIncreased;
    int rank;
  public:
    int prio;
    int deleteCount;
    std::list<int> bringInList;  
    std::vector<CkIOMetaData *> dependences;
    bool dead;
    
    bool checkReady(){
      bool ready = true;
      for(int i = 0; i < dependences.size(); i++){
        if(dependences[i]->getState() != INMEM){
          ready = false;
          break;
        }
      }
      if(ready){
        if(!refCountIncreased)
          incRefCount();
        if(CkOOCMode == OOC_ADAPTIVE){
          //push to the shared node queue
          CmiLock(obj->pendingQLock);
          bool shouldPush = (!obj->inUse)&&(obj->pendingQ.size() == 0);
          obj->pendingQ.push(new CkTask(epIdx, msg, doFree));
          CmiUnlock(obj->pendingQLock);
          if(shouldPush){
            CmiLock(objQLock);
            objQ.push(obj);
            CmiUnlock(objQLock);
          }
        }
        else{
          privateObjQ[rank].push(new CkTask(epIdx, msg, obj, doFree));;
        }
      }
      return ready;
    }

    void addToList(){
      for(int i = 0; i < dependences.size(); i++){
        if(dependences[i]->getState() == INDISK){
          bringInList.push_back(dependences[i]->getID());
        }
        dependences[i]->addTask(this);
      }
    }

    OOCTask(std::vector<CkIOMetaData *> & params, void * _obj, int _epIdx, char * _msg, bool _doFree){
      dependences = params;
      obj = (CkMigratable *)_obj;
      epIdx = _epIdx;
      msg = _msg;
      doFree = _doFree;
      rank = CmiMyRank();
      
      prio = 0;
      if(UsrToEnv(msg)->getPriobits() != 0)
        prio = *((int*)CkPriorityPtr(msg));
      
      deleteCount = 0;
      dead = false;
      refCountIncreased = false;
    }
    
    void incRefCount(){
      refCountIncreased = true;
      dead = true;//should remove from pending tasks
      for(int i = 0; i < dependences.size(); i++){
        dependences[i]->incRefCount();
      }
    }
};

void scheduleTask(OOCTask * task);

static void OOCInvokeCompute(std::vector<CkIOMetaData *> params, void * obj, int epIdx, char * msg, bool doFree){
  OOCTask * task = new OOCTask(params, obj, epIdx, msg, doFree);
  scheduleTask(task);
}

static void OOCComputeDone(std::vector<CkIOMetaData *> params){
  pthread_mutex_lock(&IOLock);
  for(int i = 0; i < params.size(); i++){
    params[i]->decRefCount();
  }
  if(!IOSchedulerActive){
    pthread_cond_signal(&IOSig);
  }
  pthread_mutex_unlock(&IOLock);
}

CpvExtern(std::vector<CkIOMetaData *>, oocParams);

template <typename T>
static void addOOCDep(CkIOHandle<T> * oneParam){
  CpvAccess(oocParams).push_back(oneParam->getMD());
}

template <typename T>
static void releaseOOCDep(CkIOHandle<T> * oneParam){
  CpvAccess(oocParams).push_back(oneParam->getMD());
}

static void addAllOOCDeps(void* obj, void * msg, int epIdx, bool doFree){
  OOCInvokeCompute(CpvAccess(oocParams), obj, epIdx, (char *)msg, doFree);
  CpvAccess(oocParams).clear();
}

static void releaseAllOOCDeps(){
  OOCComputeDone(CpvAccess(oocParams));
  CpvAccess(oocParams).clear();
}

void OOCWriteAll();
void OOCInit(long size, std::string prefix, int oocMode);
#endif
