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

void InfoStruct::set(const char* k, const char* v){
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
}

void InfoStruct::dup(InfoStruct& src){
  int sz=src.nodes.size();
  for(int i=0;i<sz;i++){
    KeyvalPair* newkvp = new KeyvalPair(src.nodes[i]->key,src.nodes[i]->val);
    nodes.push_back(newkvp);
  }
}

int InfoStruct::deletek(const char* k){
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      delete nodes[i];
      nodes.remove(i);
      found=1;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get(const char* k, int vl, char*& v){
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      strncpy(v, nodes[i]->val, vl);
      if(vl<strlen(nodes[i]->val)) v[vl]='\0';
      found=1;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get_valuelen(const char* k, int* vl){
  const char *key = create_stripped_string(k);
  int sz=nodes.size();
  int found=0;
  for(int i=0;i<sz;i++){
    if(!strcmp(nodes[i]->key, key)){
      *vl=strlen(nodes[i]->val);
      found=1;
      break;
    }
  }
  delete [] key;
  return found;
}

int InfoStruct::get_nthkey(int n,char* k){
  if(n<0 || n>=nodes.size())
    return 0;
  strcpy(k,nodes[n]->key);
  return 1;
}

void InfoStruct::myfree(void){
  int sz=nodes.size();
  for(int i=0;i<sz;i++){
    delete nodes[i];
  }
  nodes.resize(0);
  valid=false;
}

MPI_Info ampiParent::createInfo(void){
  InfoStruct* newinfo = new InfoStruct;
  infos.push_back(newinfo);
  return (MPI_Info)(infos.size()-1);
}

MPI_Info ampiParent::dupInfo(MPI_Info info){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_dup: invalid info\n");
  InfoStruct* newinfo = new InfoStruct;
  newinfo->dup(*infos[info]);
  infos.push_back(newinfo);
  return (MPI_Info)(infos.size()-1);
}

void ampiParent::setInfo(MPI_Info info, const char *key, const char *value){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_set: invalid info\n");
  infos[info]->set(key,value);
}

int ampiParent::deleteInfo(MPI_Info info, const char *key){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_delete: invalid info\n");
  return infos[info]->deletek(key);
}

int ampiParent::getInfo(MPI_Info info, const char *key, int valuelen, char *value){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_get: invalid info\n");
  return infos[info]->get(key,valuelen,value);
}

int ampiParent::getInfoValuelen(MPI_Info info, const char *key, int *valuelen){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_get_valuelen: invalid info\n");
  return infos[info]->get_valuelen(key,valuelen);
}

int ampiParent::getInfoNkeys(MPI_Info info){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_get_nkeys: invalid info\n");
  return infos[info]->get_nkeys();
}

int ampiParent::getInfoNthkey(MPI_Info info, int n, char *key){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_get_nthkey: invalid info\n");
  return infos[info]->get_nthkey(n,key);
}

void ampiParent::freeInfo(MPI_Info info){
  if(info<0 || info>=infos.size() || !infos[info]->getvalid())
    CkAbort("AMPI_Info_free: invalid info\n");
  infos[info]->myfree();
}


CDECL
int AMPI_Info_create(MPI_Info *info){
  AMPIAPI("AMPI_Info_create");
  if(info<=(int *)0)
    CkAbort("AMPI_Info_create: invalid info\n");
  ampiParent *ptr = getAmpiParent();
  *info = ptr->createInfo();
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_set(MPI_Info info, const char *key, const char *value){
  AMPIAPI("AMPI_Info_set");
  if(key<=(char *)0 || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    CkAbort("AMPI_Info_set: invalid key\n");
  if(value<=(char *)0 || strlen(value)>MPI_MAX_INFO_VAL || strlen(value)==0)
    CkAbort("AMPI_Info_set: invalid value\n");
  ampiParent *ptr = getAmpiParent();
  ptr->setInfo(info, key, value);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_delete(MPI_Info info, const char *key){
  AMPIAPI("AMPI_Info_delete");
  ampiParent *ptr = getAmpiParent();
  if(key<=(char *)0 || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    CkAbort("AMPI_Info_delete: invalid key\n");
  if(0==ptr->deleteInfo(info, key))
    CkAbort("AMPI_Info_delete: key not defined in info\n");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag){
  AMPIAPI("AMPI_Info_get");
  if(key<=(char *)0 || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    CkAbort("AMPI_Info_get: invalid key\n");
  if(value<=(char *)0)
    CkAbort("AMPI_Info_get: invalid value\n");
  if(valuelen<=0)
    CkAbort("AMPI_Info_get: invalid valuelen\n");
  ampiParent *ptr = getAmpiParent();
  *flag = ptr->getInfo(info, key, valuelen, value);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag){
  AMPIAPI("AMPI_Info_get_valuelen");
  if(key<=(char *)0 || strlen(key)>MPI_MAX_INFO_KEY || strlen(key)==0)
    CkAbort("AMPI_Info_get_valuelen: invalid key\n");
  ampiParent *ptr = getAmpiParent();
  *flag = ptr->getInfoValuelen(info, key, valuelen);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_get_nkeys(MPI_Info info, int *nkeys){
  AMPIAPI("AMPI_Info_get_nkeys");
  if(nkeys<=(int *)0)
    CkAbort("AMPI_Info_get_nkeys: invalid nkeys\n");
  ampiParent *ptr = getAmpiParent();
  *nkeys = ptr->getInfoNkeys(info);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_get_nthkey(MPI_Info info, int n, char *key){
  AMPIAPI("AMPI_Info_get_nthkey");
  if(key<=(char *)0)
    CkAbort("AMPI_Info_get_nthkey: invalid key\n");
  ampiParent *ptr = getAmpiParent();
  if(0==ptr->getInfoNthkey(info,n,key))
    CkAbort("AMPI_Info_get_nthkey: invalid n\n");
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_dup(MPI_Info info, MPI_Info *newinfo){
  AMPIAPI("AMPI_Info_dup");
  if(newinfo<=(int *)0)
    CkAbort("AMPI_Info_dup: invalid newinfo\n");
  ampiParent *ptr = getAmpiParent();
  *newinfo = ptr->dupInfo(info);
  return MPI_SUCCESS;
}

CDECL
int AMPI_Info_free(MPI_Info *info){
  AMPIAPI("AMPI_Info_free");
  if(info<=(int *)0)
    CkAbort("AMPI_Info_free: invalid info\n");
  ampiParent *ptr = getAmpiParent();
  ptr->freeInfo(*info);
  *info = MPI_INFO_NULL;
  return MPI_SUCCESS;
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
