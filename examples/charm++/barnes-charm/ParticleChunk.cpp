#include "barnes.h"

#include "barnes.decl.h"
void ParticleChunk::SlaveStart(bodyptr *bodystart, cellptr *cellstart, leafptr *leafstart){
  unsigned int ProcessId;

  /* Get unique ProcessId */
  ProcessId = thisIndex;

  /* POSSIBLE ENHANCEMENT:  Here is where one might pin processes to
     processors to avoid migration */

  /* initialize mybodytabs */
  mybodytab = bodystart + (maxmybody * ProcessId);
  /* note that every process has its own copy   */
  /* of mybodytab, which was initialized to the */
  /* beginning of the whole array by proc. 0    */
  /* before create                              */
  mycelltab = cellstart + (maxmycell * ProcessId);
  myleaftab = leafstart + (maxmyleaf * ProcessId);
  /* POSSIBLE ENHANCEMENT:  Here is where one might distribute the
     data across physically distributed memories as desired. 

     One way to do this is as follows:

     int i;

     if (ProcessId == 0) {
     for (i=0;i<NPROC;i++) {
     Place all addresses x such that 
     &(Local[i]) <= x < &(Local[i])+
     sizeof(struct local_memory) on node i
     Place all addresses x such that 
     &(Local[i].mybodytab) <= x < &(Local[i].mybodytab)+
     maxmybody * sizeof(bodyptr) - 1 on node i
     Place all addresses x such that 
     &(Local[i].mycelltab) <= x < &(Local[i].mycelltab)+
     maxmycell * sizeof(cellptr) - 1 on node i
     Place all addresses x such that 
     &(Local[i].myleaftab) <= x < &(Local[i].myleaftab)+
     maxmyleaf * sizeof(leafptr) - 1 on node i
     }
     }

     barrier(Global->Barstart,NPROC);

*/

  find_my_initial_bodies(bodytab, nbody, ProcessId);

  /* main loop */
  while (Local[ProcessId].tnow < tstop + 0.1 * dtime) {
    stepsystem(ProcessId);
  }
}
