
#define CsmWildCard CmmWildCard

void CsmTVSend
  CMK_PROTO((int pe, int ntags, int *tags, void *buf, int buflen));

void CsmTVRecv
  CMK_PROTO((int ntags, int *tags, void *buf, int buflen, int *rtags));

#define CsmTSend(pe, tag, buf, buflen)\
  { int CsmTag=(tag); CsmTVSend(pe, 1, &CsmTag, buf, buflen); }

#define CsmTRecv(tag, buf, buflen, rtag)\
  { int CsmTag=(tag); CsmTVRecv(1, &CsmTag, buf, buflen, rtag); }
