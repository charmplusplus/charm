#define SMWildCard CmmWildCard

void send(int pe, int ntags, int *tags, void *buf, int buflen);

void recv(int ntags, int *tags, void *buf, int buflen, int *rtags);

#define send1(pe, tag, buf, buflen)\
  { int CsmTag=(tag); send(pe, 1, &CsmTag, buf, buflen); }

#define recv1(tag, buf, buflen, rtag)\
  { int CsmTag=(tag); recv(1, &CsmTag, buf, buflen, rtag); }
