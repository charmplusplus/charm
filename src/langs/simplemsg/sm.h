#ifndef _SM_H
#define _SM_H

#define SMWildCard CmmWildCard

#ifdef __cplusplus
extern "C" {
#endif

extern void
SMInit(char**);

extern void 
GeneralSend(int pe, int ntags, int *tags, void *buf, int buflen);

extern int 
GeneralBroadcast(int rootpe, int ntags, int *tags, 
                 void *buf, int buflen, int *rtags);

extern int 
GeneralRecv(int ntags, int *tags, void *buf, int buflen, int *rtags);

#ifdef __cplusplus
}
#endif

static void send(int pe, int tag, int buflen, void *buf)
{ 
  int tags[2];
  tags[0] = CmiMyPe();
  tags[1] = (tag); 
  GeneralSend(pe, 2, tags, buf, buflen); 
}

static int broadcast(int rootpe, int tag, int buflen, void *buf, int *rtag)
{ 
  int CsmTag=(tag); 
  return GeneralBroadcast(rootpe, 1, &CsmTag, buf, buflen, rtag); 
}

static int recv(int pe, int tag, int buflen, void *buf)
{
  int tags[2];
  int rtag;
  tags[0] = pe;
  tags[1] = tag;
  return GeneralRecv(2, tags, buf, buflen, &rtag); 
}

#endif
