#ifndef _SM_H
#define _SM_H

#define SMWildCard CmmWildCard

extern void 
GeneralSend(int pe, int ntags, int *tags, void *buf, int buflen);

extern int 
GeneralBroadcast(int rootpe, int ntags, int *tags, 
                 void *buf, int buflen, int *rtags);

extern int 
GeneralRecv(int ntags, int *tags, void *buf, int buflen, int *rtags);

static void send(pe, tag, buf, buflen)
int pe, buflen;
int tag;
void *buf;
{ 
  int CsmTag=(tag); 
  GeneralSend(pe, 1, &CsmTag, buf, buflen); 
}

static int broadcast(rootpe, tag, buf, buflen, rtag)
int rootpe, buflen;
int tag, *rtag;
void *buf;
{ 
  int CsmTag=(tag); 
  return GeneralBroadcast(rootpe, 1, &CsmTag, buf, buflen, rtag); 
}

static int recv(tag, buf, buflen, rtag)
int tag, buflen;
int *rtag;
void *buf;
{
  int CsmTag=(tag); 
  return GeneralRecv(1, &CsmTag, buf, buflen, rtag); 
}

#endif
