
#define CsmWildCard CmmWildCard

void CsmTVSend(int pe, int ntags, int *tags, void *buf, int buflen);

void CsmTVRecv (int ntags, int *tags, void *buf, int buflen, int *rtags);

#define CsmTSend(pe, tag, buf, buflen)\
  do { int CsmTag=(tag); CsmTVSend(pe, 1, &CsmTag, buf, buflen); } while(0)

#define CsmTRecv(tag, buf, buflen, rtag)\
  do { int CsmTag=(tag); CsmTVRecv(1, &CsmTag, buf, buflen, rtag); } while(0)
