/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef ELEMENTS_DEFS_H
#define ELEMENTS_DEFS_H

#include "converse.h"
#include "lbdb.h"

#include "ckset.h"
#include "cklists.h"

class minHeap;
class maxHeap;

class InfoRecord
{
public:
   double load;
//   LDOMid omID;
//   LDObjid id; 
   int Id; // should replace other Ids.
};


class computeInfo : public InfoRecord
{
public: 
   /*   int computeId; replaced by Id */
//   LDObjHandle handle;
   LDObjid  id;
#ifdef TEMP_LDB
	 int omid;
#endif

   int processor; // caller to ReBalancer MAY leave this field -1, 
   int oldProcessor; // stores the current assignment of the compute object.
   int migratable;
   CkVec<int>  sendmessages;
   CkVec<int>  recvmessages;
};

class processorInfo: public InfoRecord
{
public:
   // int processorNum; replaced by inherited "Id".
   double backgroundLoad; // background work pre-assigned to the processor.
   double computeLoad;    //load due to computes. The total load is computed
                          // by adding these two.		     
   double pe_speed;
   double utilization;
   bool available;
   Set *computeSet; // caller to ReBalancer should leave this field NULL.
};

#endif

/*@}*/
