#ifndef _SOCKETREADANDWRITE_H_
#define _SOCKETREADANDWRITE_H_

int RecvSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes);
int SendSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes);

#endif
