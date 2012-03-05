#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "pvm3.h"
#include <converse.h>

/* #include <sys/types.h> */
/* #include <fcntl.h> */

#define MALLOC(x) CmiAlloc(x)
#define PRINTF CmiPrintf
#define puts(x) CmiPrintf("%s%c",(x),'\n')

pvmc_user_main(argc,argv)
{
  if (CmiMyPe()==0) {
    time_master(argc,argv);
  } else {
    time_slave(argc,argv);
  }
  exit(1);
}

/*
*       timing.c
*
*       Does a few communication timing tests on pvm.
*       Uses `timing_slave' to echo messages.
*
*       9 Dec 1991  Manchek
*  14 Oct 1992  Geist  - revision to pvm3
*/


#define SLAVENAME "timeslave"

time_master(argc, argv)
int argc;
char **argv;
{
  int mytid;                  /* my task id */
  int stid = 0;               /* slave task id */
  int reps = 20;              /* number of samples per test */
  struct timeval tv1, tv2;    /* for timing */
  int dt1, dt2;               /* time for one iter */
  int at1, at2;               /* accum. time */
  int numint;                 /* message length */
  int n;
  int i;
  int *iarray = 0;

  /* enroll in pvm */

  if ((mytid = pvm_mytid()) < 0) {
    exit(1);
  }
  PRINTF("i'm t%x\n", mytid);

  /* start up slave task */
  /*        PRINTF("Argc==%d, name == %s\n", argc, argv[1]);*/
  if (argc>1)  /* read it of the command line */
    {
      if (pvm_spawn(SLAVENAME, (char **)0, PvmTaskHost, argv[1], 1, &stid) <=0 || stid < 0) {
	fputs("can't initiate slave on ", stderr);
	fputs(argv[1], stderr);
	fputs("\n", stderr);
	goto bail;
      }
    }
  else   /* let the pvmd decide */
    if (pvm_spawn(SLAVENAME, (char**)0, 0, "", 1, &stid) <= 0 || stid < 0) {
      fputs("can't initiate slave\n", stderr);
      goto bail;
    }

  PRINTF("slave is task t%x\n", stid);

  /*
   *  round-trip timing test
   */

  puts("Doing Round Trip test, minimal message size\n");
  at1 = 0;

  /* pack buffer */

  pvm_initsend(PvmDataRaw);
  pvm_pkint(&stid, 1, 1);

  puts(" N     uSec");
  for (n = 1; n <= reps; n++) {
    gettimeofday(&tv1, (struct timezone*)0);

    if (pvm_send(stid, 1)) {
      PRINTF("can't send to \"%s\"\n", SLAVENAME);
      goto bail;
    }

    if (pvm_recv(-1, -1) < 0) {
      PRINTF("recv error%d\n" );
      goto bail;
    }

    gettimeofday(&tv2, (struct timezone*)0);

    dt1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    /*                PRINTF("(sec = %d %d, usec = %d %d) %2d %8d\n", tv2.tv_sec, tv1.tv_sec, tv2.tv_usec, tv1.tv_usec, n, dt1);*/
    PRINTF("TIME 0 %2d %8d\n", n, dt1);
    at1 += dt1;
  }
  PRINTF("RTT Avg uSec %d\n", at1 / reps);

  /*
   *  bandwidth test for different message lengths
   */

  puts("\nDoing Bandwidth tests\n");

  for (numint = 25; numint < 100000; numint *= 10) {
    PRINTF("\nMessage size is %d integers (%d bytes)\n", 
	   numint, numint * sizeof(int));
    at1 = at2 = 0;
    iarray = (int*)MALLOC(numint * sizeof(int));
    puts(" N  Pack uSec  Send uSec");
    for (n = 1; n <= reps; n++) {
      gettimeofday(&tv1, (struct timezone*)0);

      pvm_initsend(PvmDataRaw);
      pvm_pkint(iarray, numint, 1);

      gettimeofday(&tv2, (struct timezone*)0);
      dt1 = (tv2.tv_sec - tv1.tv_sec) * 1000000
	+ tv2.tv_usec - tv1.tv_usec;

      gettimeofday(&tv1, (struct timezone*)0);

      if (pvm_send(stid, 1)) {
	PRINTF("can't send to \"%s\"\n", SLAVENAME);
	goto bail;
      }

      if (pvm_recv(-1, -1) < 0) {
	PRINTF("recv error%d\n" );
	goto bail;
      }

      gettimeofday(&tv2, (struct timezone*)0);
      dt2 = (tv2.tv_sec - tv1.tv_sec) * 1000000
	+ tv2.tv_usec - tv1.tv_usec;

      PRINTF("TIME %d %2d   %8d   %8d   %8d\n", sizeof(int)*numint, n, dt1, dt2,dt1+dt2);
      at1 += dt1;
      at2 += dt2;
    }

    puts("Total uSec");
    PRINTF("     %8d   %8d   %8d\n", at1, at2,at1+at2);

    at1 /= reps;
    at2 /= reps;
    puts("Avg uSec");
    PRINTF("     %8d   %8d\n", at1, at2);
    puts("Avg Byte/uSec");
    PRINTF("     %8f   %8f\n",
	   (numint * 4) / (double)at1,
	   (numint * 4) / (double)at2);
  }

  puts("\ndone");

bail:
  if (stid > 0)
    pvm_kill(stid);
  sleep(10);
  pvm_exit();
  exit(1);
}


/*
*       timing_slave.c
*
*       See timing.c
*/

time_slave(argc, argv)
int argc;
char **argv;
{
  int mytid;   /* my task id */
  int dtid;    /* driver task */
  int bufid;
  int n = 0;

  /* enroll in pvm */

  mytid = pvm_mytid();

  /* pack mytid in buffer */

  pvm_initsend(PvmDataRaw);
  pvm_pkint(&mytid, 1, 1);

  /* our job is just to echo back to the sender when we get a message */

  while (1) {
    bufid = pvm_recv(-1, -1);
    pvm_bufinfo(bufid, (int*)0, (int*)0, &dtid);
    pvm_send(dtid, 2);
    /*
      PRINTF("echo %d\n", ++n);
      */
  }
}

