#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ccs-client.h"

#define SHRINK 1
#define EXPAND 0
#define OP EXPAND

#define BUF 255

int main (int argc, char **argv)
{
    int OLDNPROCS, NEWNPROCS;

    // Create a CcsServer and connect to the given hostname and port
    CcsServer server;
    char host[BUF], *msg;
    int i, port, cmdLen, numKilled, numAdded;
    bool isExpand;

    sprintf(host, "%s", argv[1]);
    sscanf(argv[2], "%d", &port);
    sscanf(argv[3], "%d", &OLDNPROCS);
    sscanf(argv[4], "%d", &numKilled);
    int killedIndex[numKilled];
    
    for (i = 0; i < numKilled; i++) {
        sscanf(argv[5 + i], "%d", &killedIndex[i]);
    }

    sscanf(argv[5 + numKilled], "%d", &numAdded);

    NEWNPROCS = OLDNPROCS - numKilled + numAdded;

    //printf("Connecting to server %s %d\n", host, port);
    if (CcsConnect(&server, host, port, NULL) == -1) {
        printf("0");
        return 0;
    }
    //printf("Connected to server\n");

    cmdLen = 2 * sizeof(int) + OLDNPROCS * sizeof(char);
    msg = (char *) malloc(cmdLen);
    memcpy(msg, &NEWNPROCS, sizeof(int));
    memcpy(&msg[sizeof(int)], &OLDNPROCS, sizeof(int));
    
    int offset = 2 * sizeof(int);
    int count = 0;
    for (i = 0; i < OLDNPROCS; i++) {
        if (numKilled > 0 && i == killedIndex[count]) {
            msg[i + offset] = 0;
            count++;
        }
        else
            msg[i + offset] = 1;
    }

    for (i = 0; i < OLDNPROCS; i++) {
        printf("PE %d: %d\n", i, msg[i + offset]);
    }

    //memcpy(&msg[sizeof(bool)], &NEWNPROCS, sizeof(int));
    CcsSendRequest(&server, "set_bitmap", 0, cmdLen, msg);

    //printf("Waiting for reply...\n" );
    //CcsRecvResponse(&server, cmdLen, msg , 180);
    //printf("Reply received.\n");

    return 0;
}
