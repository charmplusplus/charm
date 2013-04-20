/*************************************************************************/
/*                                                                       */
/*  Copyright (c) 1994 Stanford University                               */
/*                                                                       */
/*  All rights reserved.                                                 */
/*                                                                       */
/*  Permission is given to use, copy, and modify this software for any   */
/*  non-commercial purpose as long as this copyright notice is not       */
/*  removed.  All other uses, including redistribution in whole or in    */
/*  part, are forbidden without prior written permission.                */
/*                                                                       */
/*  This software is provided with absolutely no warranty and no         */
/*  support.                                                             */
/*                                                                       */
/*************************************************************************/

/*
 * CODE_IO.C: 
 */
 
#include "barnes.h"

 /*
#include <stdlib.h> 
#include <semaphore.h> 
#include <assert.h> 
 */
    
    
/*
 * INPUTDATA: read initial conditions from input file.
 */
    
void Main::inputdata ()
{
   std::ifstream instr;
   char headbuf[128];
   int ndim,counter=0;
   bodyptr p;
   int i;
   fprintf(stdout,"reading input file : %s\n",infile.c_str());
   fflush(stdout);
   //instr = fopen(infile, "r");
   instr.open(infile.c_str());
   /*
   if (instr == NULL)
      error("inputdata: cannot find file %s\n", infile);
      */
   //in_int(instr, &nbody);
   instr >> nbody;
   if (nbody < 1){
      CkPrintf("inputdata: nbody = %d is absurd\n", nbody);
      CkAbort("");
   }

   //in_int(instr, &ndim);
   instr >> ndim;
   if (ndim != NDIM){
      CkPrintf("inputdata: NDIM = %d ndim = %d is absurd\n", NDIM,ndim);
      CkAbort("");
   }
   
   //in_real(instr, &tnow);
   instr >> tnow;
   CkPrintf("read tnow: %f\n", tnow);
   /*
   for (i = 0; i < MAX_PROC; i++) {
      Local[i].tnow = tnow;
   }
   */

   bodytab = new body [nbody]; 
   //bodytab = (bodyptr) malloc(nbody * sizeof(body));
   //bodytab = (bodyptr) our_malloc(nbody * sizeof(body),__FILE__,__LINE__);
   if (bodytab == NULL){
      CkPrintf("inputdata: not enuf memory\n");
      CkAbort("");
   }

   for (p = bodytab; p < bodytab+nbody; p++) {
      Type(p) = BODY;
      Cost(p) = 1;
      Phi(p) = 0.0;
      CLRV(Acc(p));
   }
#ifdef OUTPUT_ACC
   int seq = 0;
#endif
   for (p = bodytab; p < bodytab+nbody; p++){
      instr >> Mass(p);
#ifdef OUTPUT_ACC
      p->num = seq;
      seq++;
#endif
   }
   for (p = bodytab; p < bodytab+nbody; p++){
      instr >> Pos(p)[0];
      instr >> Pos(p)[1];
      instr >> Pos(p)[2];
   }
   for (p = bodytab; p < bodytab+nbody; p++){
      instr >> Vel(p)[0];
      instr >> Vel(p)[1];
      instr >> Vel(p)[2];
   }
   instr.close();
}
