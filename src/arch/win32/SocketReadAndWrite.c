#include "windows.h"
#include "stdio.h"
#include "winsock.h"
#include <io.h>

int RecvSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes)
{
	int nLeft;
	int nRead;
	int nTotal = 0;

	nLeft = nBytes;
	while (0 < nLeft)
	{
		nRead = recv(hSocket,pBuff,nLeft,0);
		if (SOCKET_ERROR == nRead)
		{
			return nRead;
		}
		else
		{
			nLeft -= nRead;
			pBuff += nRead;
			nTotal += nRead;
		}
	}

	return nTotal;
}

int SendSocketN(SOCKET hSocket,BYTE *pBuff,int nBytes)
{
	int nLeft,nWritten;
	int nTotal = 0;

	nLeft = nBytes;
	while (0 < nLeft)
	{
		nWritten = send(hSocket,pBuff,nLeft,0);
		if (SOCKET_ERROR == nWritten)
		{
			return nWritten;
		}
		else
		{
			nLeft -= nWritten;
			pBuff += nWritten;
			nTotal += nWritten;
		}
	}
	return nTotal;
}
