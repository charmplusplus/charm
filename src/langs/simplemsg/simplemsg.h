
#define CsmWildCard CmmWildCard

void CsmTVSend(int pe, int ntags, int *tags, void *buf, int buflen);

void CsmTVRecv (int ntags, int *tags, void *buf, int buflen, int *rtags);

#define CsmTSend(pe, tag, buf, buflen)\
  { int CsmTag=(tag); CsmTVSend(pe, 1, &CsmTag, buf, buflen); }

#define CsmTRecv(tag, buf, buflen, rtag)\
  { int CsmTag=(tag); CsmTVRecv(1, &CsmTag, buf, buflen, rtag); }
