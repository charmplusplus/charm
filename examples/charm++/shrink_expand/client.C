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

    if (argc < 5) {
        printf("Usage: %s <hostname> <port> <oldprocs> <newprocs> \n", argv[0]);
        return 1;
    }

    // Create a CcsServer and connect to the given hostname and port
    CcsServer server;
    char host[BUF], *msg;
    int i, port, cmdLen;
    bool isExpand;

    sprintf(host, "%s", argv[1]);
    sscanf(argv[2], "%d", &port);
    sscanf(argv[3], "%d", &OLDNPROCS);
    sscanf(argv[4], "%d", &NEWNPROCS);

    if( NEWNPROCS > OLDNPROCS)
        isExpand = true;
    else if(OLDNPROCS > NEWNPROCS)
        isExpand = false;
    else{
        printf("1");
        return 0;
    }
    //printf("Connecting to server %s %d\n", host, port);
    if (CcsConnect(&server, host, port, NULL) == -1) {
        printf("0");
        return 0;
    }
    //printf("Connected to server\n");

    cmdLen = sizeof(int) + sizeof(bool);
    msg = (char *) malloc(cmdLen);
    memcpy(msg, &isExpand, sizeof(bool));
    memcpy(&msg[sizeof(bool)], &NEWNPROCS, sizeof(int));
    CcsSendRequest(&server, "realloc", 0, cmdLen, msg);

    printf("Waiting for reply...\n" );
    CcsRecvResponse(&server, cmdLen, msg , 180);
    printf("Reply received.\n");

    return 0;
}
