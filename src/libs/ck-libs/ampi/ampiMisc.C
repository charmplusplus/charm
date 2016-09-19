/****************************************************
 * File: ampiMisc.C
 *       This file contains miscellaneous functions
 ****************************************************/
#include <string.h>
#include "ampiimpl.h"

// Strip leading/trailing whitespaces from MPI_Info key and value strings.
char* create_stripped_string(const char *str) {
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

  char *newStr = new char[newLen+1];

  if(newLen>0)
    memcpy(newStr, &str[start], newLen);
  newStr[newLen] = '\0';
  return newStr;
}

KeyvalPair::KeyvalPair(const char* k, const char* v){
  key = create_stripped_string(k);
  val = create_stripped_string(v);
  klen = strlen(key);
  vlen = strlen(val);
}

KeyvalPair::~KeyvalPair(void){
  free((char*)key);
  free((char*)val);
}

void InfoStruct::pup(PUP::er& p){
  p|nodes;
  p|valid;
}

int InfoStruct::set(const char* k, const char* v){
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      free((char*)(nodes[i]->val));
      nodes[i]->val = create_stripped_string(v);
      found=1;
      break;
    }
  }
  if(!found){
    KeyvalPair* newkvp = new KeyvalPair(k,v);
    nodes.push_back(newkvp);
  }
  delete [] key;
  return MPI_SUCCESS;
}

int InfoStruct::dup(InfoStruct& src){
  int sz=src.nodes.size();
  for(int i=0;i<sz;i++){
    KeyvalPair* newkvp = new KeyvalPair(src.nodes[i]->key,src.nodes[i]->val);
    nodes.push_back(newkvp);
  }
  return MPI_SUCCESS;
}

int InfoStruct::deletek(const char* k){
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=MPI_ERR_INFO_KEY;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      delete nodes[i];
      nodes.remove(i);
      found=MPI_SUCCESS;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get(const char* k, int vl, char*& v, int *flag) const{
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=MPI_ERR_INFO_KEY;
  *flag=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      strncpy(v, nodes[i]->val, vl);
      if(vl<strlen(nodes[i]->val)) v[vl]='\0';
      found=MPI_SUCCESS;
      *flag=1;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get_valuelen(const char* k, int* vl, int *flag) const{
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=MPI_ERR_INFO_KEY;
  *flag=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      *vl=strlen(nodes[i]->val);
      found=MPI_SUCCESS;
      *flag=1;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get_nkeys(int *n) const{
  *n = nodes.size();
  return MPI_SUCCESS;
}

int InfoStruct::get_nthkey(int n, char* k) const{
#if AMPI_ERROR_CHECKING
  if(n<0 || n>=nodes.size())
    return MPI_ERR_INFO_KEY;
#endif
  strcpy(k,nodes[n]->key);
  return MPI_SUCCESS;
}

void InfoStruct::myfree(void){
  int sz=nodes.size();
  for(int i=0;i<sz;i++){
    delete nodes[i];
  }
  nodes.resize(0);
  valid=false;
}

int ampiParent::createInfo(MPI_Info *newinfo){
#if AMPI_ERROR_CHECKING
  if(newinfo==NULL)
    return MPI_ERR_INFO;
#endif
  InfoStruct* newInfoStruct = new InfoStruct;
  infos.push_back(newInfoStruct);
  *newinfo = (MPI_Info)(infos.size()-1);
  return MPI_SUCCESS;
}

int ampiParent::dupInfo(MPI_Info info, MPI_Info *newinfo){
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

int ampiParent::setInfo(MPI_Info info, const char *key, const char *value){
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

int ampiParent::deleteInfo(MPI_Info info, const char *key){
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->deletek(key);
}

int ampiParent::getInfo(MPI_Info info, const char *key, int valuelen, char *value, int *flag) const{
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

int ampiParent::getInfoValuelen(MPI_Info info, const char *key, int *valuelen, int *flag) const{
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->get_valuelen(key,valuelen,flag);
}

int ampiParent::getInfoNkeys(MPI_Info info, int *nkeys) const{
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(nkeys==NULL)
    return MPI_ERR_ARG;
#endif
  return infos[info]->get_nkeys(nkeys);
}

int ampiParent::getInfoNthkey(MPI_Info info, int n, char *key) const{
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
  if(key==NULL)
    return MPI_ERR_INFO_KEY;
#endif
  return infos[info]->get_nthkey(n,key);
}

int ampiParent::freeInfo(MPI_Info info){
#if AMPI_ERROR_CHECKING
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    return MPI_ERR_INFO;
#endif
  infos[info]->myfree();
  info = MPI_INFO_NULL;
  return MPI_SUCCESS;
}


CDECL
int AMPI_Info_create(MPI_Info *info){
  AMPIAPI("AMPI_Info_create");
  int ret = getAmpiParent()->createInfo(info);
  return ampiErrhandler("AMPI_Info_create", ret);
}

CDECL
int AMPI_Info_set(MPI_Info info, const char *key, const char *value){
  AMPIAPI("AMPI_Info_set");
  int ret = getAmpiParent()->setInfo(info, key, value);
  return ampiErrhandler("AMPI_Info_set", ret);
}

CDECL
int AMPI_Info_delete(MPI_Info info, const char *key){
  AMPIAPI("AMPI_Info_delete");
  int ret = getAmpiParent()->deleteInfo(info, key);
  return ampiErrhandler("AMPI_Info_delete", ret);
}

CDECL
int AMPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag){
  AMPIAPI("AMPI_Info_get");
  int ret = getAmpiParent()->getInfo(info, key, valuelen, value, flag);
  return ampiErrhandler("AMPI_Info_get", ret);
}

CDECL
int AMPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag){
  AMPIAPI("AMPI_Info_get_valuelen");
  int ret = getAmpiParent()->getInfoValuelen(info, key, valuelen, flag);
  return ampiErrhandler("AMPI_Info_get_valuelen", ret);
}

CDECL
int AMPI_Info_get_nkeys(MPI_Info info, int *nkeys){
  AMPIAPI("AMPI_Info_get_nkeys");
  int ret = getAmpiParent()->getInfoNkeys(info, nkeys);
  return ampiErrhandler("AMPI_Info_get_nkeys", ret);
}

CDECL
int AMPI_Info_get_nthkey(MPI_Info info, int n, char *key){
  AMPIAPI("AMPI_Info_get_nthkey");
  int ret = getAmpiParent()->getInfoNthkey(info, n, key);
  return ampiErrhandler("AMPI_Info_get_nthkey", ret);
}

CDECL
int AMPI_Info_dup(MPI_Info info, MPI_Info *newinfo){
  AMPIAPI("AMPI_Info_dup");
  int ret = getAmpiParent()->dupInfo(info, newinfo);
  return ampiErrhandler("AMPI_Info_dup", ret);
}

CDECL
int AMPI_Info_free(MPI_Info *info){
  AMPIAPI("AMPI_Info_free");
  int ret = getAmpiParent()->freeInfo(*info);
  *info = MPI_INFO_NULL;
  return ampiErrhandler("AMPI_Info_free", ret);
}

#ifdef AMPIMSGLOG
#if CMK_PROJECTIONS_USE_ZLIB
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
