/*
Classes for easy, transparent use of on-the-wire
network byte order integers and doubles.

Orion Sky Lawlor, olawlor@acm.org, 6/22/2001
*/
#include <stdio.h>
#include <stdlib.h>
#include "networkVar.h"

//Check the machine's endianness only once
int networkDouble::g_doFlip(networkDouble::getFlip());

int networkDouble::getFlip(void)
{
     double dtest=-9.5;//Double test value
     unsigned char *c=(unsigned char *)&dtest;
     if (c[0]==0xc0 && c[1]==0x23 && c[2]==0x00 && c[3]==0x00) 
       return 0;//Big-endian IEEE machine (e.g., Mac, Sun, SGI)
     if (c[4]==0x00 && c[5]==0x00 && c[6]==0x23 && c[7]==0xc0) 
       return 1;//Little-endian IEEE machine (e.g., Intel)
     else {
       fprintf(stderr,"ERROR! Unrecognized floating-point format!\n");
       abort();
       return 99;//<- for whining compilers
     }
}






