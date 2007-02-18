/*
 
   Simple Explicit FEA example using
      the ParFUM Tops-like API

               By 
           Isaac Dooley

 */

//#include "pgm.h"
#include "ParFUM_TOPS.h"

class ElemAtt {
public:
  int A[5];
};

class NodeAtt {
public:
  double A[10];
  int B;
};



extern "C" void
init(void)
{

  TopModel *m = topModel_Create_Init(sizeof(ElemAtt), sizeof(NodeAtt));



	{
	  TopElemItr* e_itr = topModel_CreateElemItr(m);
	  int elem_count=0;
	  for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
		elem_count++;
		// TopNode node = topElemItr_GetCurr(itr);
	  }
	  printf("init : elem_count = %d\n", elem_count);
	}

	for(int i=1;i<=7;++i){
	  TopNode node = topModel_InsertNode(m,i,i,i);
	  NodeAtt a;
	  a.A[3]=i;
	  a.B=777;
	  topNode_SetAttrib(m,node,&a);
	}
	printf("init : Inserted 7 nodes\n");

	for(int i=1;i<=17;++i){
	  TopNode conn[3] = {1,2,3};
	  TopElement elem = topModel_InsertElem(m,FEM_TRIANGULAR,conn);

	  ElemAtt a;
	  a.A[1]=i;
	  topElement_SetAttrib(m,elem,&a);

	}
	printf("init : Inserted 17 elements\n");


}


// A driver() function 
// driver() is required in all FEM programs
extern "C" void
driver(void)
{

  int myId = FEM_My_partition();
  TopModel *m = topModel_Create_Driver(sizeof(ElemAtt), sizeof(NodeAtt));


  printf("vp %d: Entering driver()\n", myId); 

	{
	  TopNodeItr* itr = topModel_CreateNodeItr(m);
	  int node_count=0;
	  for(topNodeItr_Begin(itr);topNodeItr_IsValid(itr);topNodeItr_Next(itr)){
		node_count++;
		// TopNode node = topNodeItr_GetCurr(itr);
	  }
	  printf("vp %d: node_count = %d\n", myId, node_count);
	}


	{
	  TopElemItr* e_itr = topModel_CreateElemItr(m);
	  int elem_count=0;
	  for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
		elem_count++;
		// TopNode node = topElemItr_GetCurr(itr);
	  }
	  printf("vp %d: elem_count = %d\n", myId, elem_count);
	}

}

