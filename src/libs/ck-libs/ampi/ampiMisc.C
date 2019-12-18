/****************************************************
 * File: ampiMisc.C
 *       This file contains miscellaneous functions
 ****************************************************/
#include <string.h>
#include "ampiimpl.h"

#if defined(_WIN32)
#include <windows.h>
#include <sstream>
#include <direct.h>
#include <stdlib.h>
#define getcwd _getcwd
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

// Strip leading/trailing whitespaces from MPI_Info key and value strings.
std::string create_stripped_string(const char *str) noexcept {
  int newLen;
  int len = strlen(str);

  int start=0;
  while((start<len) && (str[start]==' ')){
    start++;
  }

  if(start>=len){
    newLen=0;
  }else{
    int end=len-1;
    while((end>start) && (str[end]==' ')){
      end--;
    }
    newLen = end - start+1;
  }

  if(newLen==0){
    return std::string();
  }else{
    std::string newStr(&str[start], newLen);
    return newStr;
  }
}

KeyvalPair::KeyvalPair(const char* k, const char* v) noexcept {
  key = create_stripped_string(k);
  val = create_stripped_string(v);
}

void InfoStruct::pup(PUP::er& p) noexcept {
  p|nodes;
  p|valid;
}

int InfoStruct::set(const char* k, const char* v) noexcept {
  std::string key = create_stripped_string(k);
  for(int i=0;i<nodes.size();i++){
    if(nodes[i]->key == key){
      nodes[i]->val.clear();
      nodes[i]->val = create_stripped_string(v);
      return MPI_SUCCESS;
    }
  }

  KeyvalPair* newkvp = new KeyvalPair(k,v);
  nodes.push_back(newkvp);
  return MPI_SUCCESS;
}

int InfoStruct::dup(InfoStruct& src) noexcept {
  int nkeys = src.nodes.size();
  nodes.resize(nkeys);
  for(int i=0;i<nkeys;i++){
    nodes[i] = new KeyvalPair(src.nodes[i]->key.c_str(), src.nodes[i]->val.c_str());
  }
  return MPI_SUCCESS;
}

int InfoStruct::deletek(const char* k) noexcept {
  std::string key = create_stripped_string(k);
  for(int i=0;i<nodes.size();i++){
    if(nodes[i]->key == key){
      delete nodes[i];
      nodes.remove(i);
      return MPI_SUCCESS;
    }
  }
  return MPI_ERR_INFO_KEY;
}

int InfoStruct::get(const char* k, int vl, char*& v, int *flag) const noexcept {
  std::string key = create_stripped_string(k);
  for(int i=0;i<nodes.size();i++){
    if(nodes[i]->key == key){
      strncpy(v, nodes[i]->val.c_str(), vl);
      if(vl<strlen(nodes[i]->val.c_str()))
        v[vl]='\0';
      *flag=1;
      return MPI_SUCCESS;
    }
  }
  *flag=0;
  return MPI_ERR_INFO_KEY;
}

int InfoStruct::get_valuelen(const char* k, int* vl, int *flag) const noexcept {
  std::string key = create_stripped_string(k);
  for(int i=0;i<nodes.size();i++){
    if(nodes[i]->key == key){
      *vl=strlen(nodes[i]->val.c_str());
      *flag=1;
      return MPI_SUCCESS;
    }
  }
  *flag=0;
  return MPI_ERR_INFO_KEY;
}

int InfoStruct::get_nkeys(int *n) const noexcept {
  *n = nodes.size();
  return MPI_SUCCESS;
}

int InfoStruct::get_nthkey(int n, char* k) const noexcept {
#if AMPI_ERROR_CHECKING
  if(n<0 || n>=nodes.size())
    return MPI_ERR_INFO_KEY;
#endif
  strcpy(k,nodes[n]->key.c_str());
  return MPI_SUCCESS;
}

void InfoStruct::myfree() noexcept {
  for(int i=0;i<nodes.size();i++){
    delete nodes[i];
  }
  nodes.resize(0);
  valid=false;
}

int ampiParent::createInfo(MPI_Info *newinfo) noexcept {
#if AMPI_ERROR_CHECKING
  if(newinfo==NULL)
    return MPI_ERR_INFO;
#endif
  InfoStruct* newInfoStruct = new InfoStruct;
  infos.push_back(newInfoStruct);
  *newinfo = (MPI_Info)(infos.size()-1);
  return MPI_SUCCESS;
}

int ampiParent::dupInfo(MPI_Info info, MPI_Info *newinfo) noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(newinfo==NULL)
    return MPI_ERR_INFO;
#endif
  InfoStruct *newInfoStruct = new InfoStruct;
  newInfoStruct->dup(*infos[info]);
  infos.push_back(newInfoStruct);
  *newinfo = (MPI_Info)(infos.size()-1);
  return MPI_SUCCESS;
}

int ampiParent::setInfo(MPI_Info info, const char *key, const char *value) noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
  if(value==NULL || strlen(value)>MPI_MAX_INFO_VAL || strlen(value)==0)
    return MPI_ERR_INFO_VALUE;
#endif
  return infos[info]->set(key,value);
}

int ampiParent::deleteInfo(MPI_Info info, const char *key) noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->deletek(key);
}

int ampiParent::getInfo(MPI_Info info, const char *key, int valuelen, char *value, int *flag) const noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
  if(value==NULL)
    return MPI_ERR_INFO_VALUE;
  if(valuelen<0)
    return MPI_ERR_ARG;
#endif
  return infos[info]->get(key,valuelen,value,flag);
}

int ampiParent::getInfoValuelen(MPI_Info info, const char *key, int *valuelen, int *flag) const noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->get_valuelen(key,valuelen,flag);
}

int ampiParent::getInfoNkeys(MPI_Info info, int *nkeys) const noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(nkeys==NULL)
    return MPI_ERR_ARG;
#endif
  return infos[info]->get_nkeys(nkeys);
}

int ampiParent::getInfoNthkey(MPI_Info info, int n, char *key) const noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->get_nthkey(n,key);
}

int ampiParent::freeInfo(MPI_Info info) noexcept {
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
#endif
  infos[info]->myfree();
  info = MPI_INFO_NULL;
  return MPI_SUCCESS;
}

void ampiParent::defineInfoEnv(int nRanks_) noexcept {
  char **p_argv;
  std::string argv_str, maxprocs_str;
  char hostname[255], work_dir[1024];
#if defined(_WIN32)
  SYSTEM_INFO sys_info_data;
#else
  struct utsname uname_data;
#endif
  MPI_Info envInfo;

  p_argv = CkGetArgv();
  createInfo(&envInfo);
  CkAssert(envInfo == MPI_INFO_ENV);

  setInfo(envInfo, "command", p_argv[0]);

  for(int i=1; i<CkGetArgc(); i++) {
    argv_str += p_argv[i];
    argv_str += " ";
  }
  setInfo(envInfo, "argv", argv_str.c_str());

  maxprocs_str = std::to_string(nRanks_);
  setInfo(envInfo, "maxprocs", maxprocs_str.c_str());

  //TODO: soft

  gethostname(hostname, sizeof(hostname));
  setInfo(envInfo, "host", hostname);

  //extract arch(machine) info
#if defined(_WIN32)
  std::stringstream win_arch;
  GetSystemInfo(&sys_info_data);
  win_arch << sys_info_data.dwProcessorType;
  setInfo(envInfo, "arch", win_arch.str().c_str());
#else
  if (uname(&uname_data) == -1) {
    CkAbort("uname call in defineInfoEnv() failed\n");
  }
  else {
    setInfo(envInfo, "arch", uname_data.machine);
  }
#endif

  if (getcwd(work_dir, sizeof(work_dir)) == NULL) {
    CkAbort("AMPI> call to getcwd() for MPI_INFO_ENV failed!");
  }
  setInfo(envInfo, "wdir", work_dir);

  //TODO: file, thread_level
}

void ampiParent::defineInfoMigration() noexcept {
  MPI_Info lb_sync_info, lb_async_info, chkpt_mem_info, chkpt_msg_log_info;

  // info object for AMPI_INFO_LB_SYNC
  createInfo(&lb_sync_info);
  CkAssert(lb_sync_info == AMPI_INFO_LB_SYNC);
  setInfo(lb_sync_info, "ampi_load_balance", "sync");

  // info object for AMPI_INFO_LB_ASYNC
  createInfo(&lb_async_info);
  CkAssert(lb_async_info == AMPI_INFO_LB_ASYNC);
  setInfo(lb_async_info, "ampi_load_balance", "async");

  // info object for AMPI_INFO_CHKPT_IN_MEMORY
  createInfo(&chkpt_mem_info);
  CkAssert(chkpt_mem_info == AMPI_INFO_CHKPT_IN_MEMORY);
  setInfo(chkpt_mem_info, "ampi_checkpoint", "in_memory");
}

AMPI_API_IMPL(int, MPI_Info_create, MPI_Info *info)
{
  AMPI_API("AMPI_Info_create", info);
  int ret = getAmpiParent()->createInfo(info);
  return ampiErrhandler("AMPI_Info_create", ret);
}

AMPI_API_IMPL(int, MPI_Info_set, MPI_Info info, const char *key, const char *value)
{
  AMPI_API("AMPI_Info_set", info, key, value);
  int ret = getAmpiParent()->setInfo(info, key, value);
  return ampiErrhandler("AMPI_Info_set", ret);
}

AMPI_API_IMPL(int, MPI_Info_delete, MPI_Info info, const char *key)
{
  AMPI_API("AMPI_Info_delete", info, key);
  int ret = getAmpiParent()->deleteInfo(info, key);
  return ampiErrhandler("AMPI_Info_delete", ret);
}

AMPI_API_IMPL(int, MPI_Info_get, MPI_Info info, const char *key, int valuelen,
                                 char *value, int *flag)
{
  AMPI_API("AMPI_Info_get", info, key, valuelen, value, flag);
  getAmpiParent()->getInfo(info, key, valuelen, value, flag);
  return MPI_SUCCESS; // It is not an error if the requested key does not exist
}

AMPI_API_IMPL(int, MPI_Info_get_valuelen, MPI_Info info, const char *key,
                                          int *valuelen, int *flag)
{
  AMPI_API("AMPI_Info_get_valuelen", info, key, valuelen, flag);
  getAmpiParent()->getInfoValuelen(info, key, valuelen, flag);
  return MPI_SUCCESS; // It is not an error if the requested key does not exist
}

AMPI_API_IMPL(int, MPI_Info_get_nkeys, MPI_Info info, int *nkeys)
{
  AMPI_API("AMPI_Info_get_nkeys", info, nkeys);
  int ret = getAmpiParent()->getInfoNkeys(info, nkeys);
  return ampiErrhandler("AMPI_Info_get_nkeys", ret);
}

AMPI_API_IMPL(int, MPI_Info_get_nthkey, MPI_Info info, int n, char *key)
{
  AMPI_API("AMPI_Info_get_nthkey", info, n, key);
  int ret = getAmpiParent()->getInfoNthkey(info, n, key);
  return ampiErrhandler("AMPI_Info_get_nthkey", ret);
}

AMPI_API_IMPL(int, MPI_Info_dup, MPI_Info info, MPI_Info *newinfo)
{
  AMPI_API("AMPI_Info_dup", info, newinfo);
  int ret = getAmpiParent()->dupInfo(info, newinfo);
  return ampiErrhandler("AMPI_Info_dup", ret);
}

AMPI_API_IMPL(int, MPI_Info_free, MPI_Info *info)
{
  AMPI_API("AMPI_Info_free", info);
  int ret = getAmpiParent()->freeInfo(*info);
  *info = MPI_INFO_NULL;
  return ampiErrhandler("AMPI_Info_free", ret);
}

#ifdef AMPIMSGLOG
#if CMK_USE_ZLIB
/*zDisk PUP::er's*/
void PUP::tozDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/)
{ gzwrite(F,p,itemSize*n);}
void PUP::fromzDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/)
{ gzread(F,p,itemSize*n);}

/*zDisk buffer seeking is also simple*/
void PUP::zdisk::impl_startSeek(seekBlock &s) /*Begin a seeking block*/
  {s.data.loff=gztell(F);}
int PUP::zdisk::impl_tell(seekBlock &s) /*Give the current offset*/
  {return (int)(gztell(F)-s.data.loff);}
void PUP::zdisk::impl_seek(seekBlock &s,int off) /*Seek to the given offset*/
  {gzseek(F,s.data.loff+off,0);}
#endif
#endif

void beginTraceBigSim(char* msg){}
void endTraceBigSim(char* msg, char* param){}
