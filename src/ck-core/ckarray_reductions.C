/*
A small library of oft-used reductions for use in the 
Array Reduction Manager.  This file is #include'd by
ckarray.C.

Parallel Programming Lab, University of Illinois at Urbana-Champaign
Orion Sky Lawlor, 11/13/1999, olawlor@acm.org

This file contains the implementations of reduction_*, which are
fuctions passed as pointers to ArrayElement::contribute().
The functions are used to reduce the contributions of a PE's
array elements, and also to (further) reduce the combined
contributions of PE's that lie deeper in the reduction spanning
tree.

A simple reduction, like sum_int, looks like this:
ArrayReductionMessage *CkReduction_sum_int(int nMsg,ArrayReductionMessage **msg)
{
	int i,ret=0;
	for (i=0;i<nMsg;i++)
		ret+=*(int *)(msg[i]->data);
	return ArrayReductionMessage::buildNew(sizeof(int),(void *)&ret);
}

To keep the code small and easy to change, the implementations below
are built with preprocessor macros.  They also include debugging information.
*/
//#include <stdio.h>
#include <limits.h> //for MAX_INT, MIN_INT, etc.
#include <values.h> //for MAXDOUBLE, MINDOUBLE, etc.
/* #include "charm++.h" #include "reductions.h" */

//A simple debugging routine-- turn on to get status messages
#ifndef RED_DEB
#define RED_DEB(x) /*CkPrintf x*/
#endif

//////////////////////////// simple reductions ///////////////////
//A define used to quickly and tersely construct simple reductions
#define SIMPLE_REDUCTION(name,dataType,initRet,typeStr,loop) \
ArrayReductionMessage *name(int nMsg,ArrayReductionMessage **msg)\
{\
	RED_DEB(("/ PE_%d: " #name " invoked\n",CkMyPe()));\
	int i;\
	dataType ret=initRet;\
	for (i=0;i<nMsg;i++)\
	{\
		dataType value=*(dataType *)(msg[i]->data);\
		RED_DEB(("|\tmsg[%d (from %d)]="typeStr"\n",i,msg[i]->source,value));\
		loop\
	}\
	RED_DEB(("\\ PE_%d: " #name " finished\n",CkMyPe()));\
	return ArrayReductionMessage::buildNew(sizeof(dataType),(void *)&ret);\
}

//Compute the sum the integers passed by each element.
SIMPLE_REDUCTION(CkReduction_sum_int,int,0,"%d",
	ret+=value;
)

//Compute the largest integer passed by any element.
SIMPLE_REDUCTION(CkReduction_max_int,int,INT_MIN,"%d",
	if (ret<value) 
		ret=value;
)

//Compute the smallest integer passed by any element.
SIMPLE_REDUCTION(CkReduction_min_int,int,INT_MAX,"%d",
	if (ret>value) 
		ret=value;
)

//Compute the logical AND of the integers passed by each element.
// The resulting integer will be zero if any source integer is zero; else 1.
SIMPLE_REDUCTION(CkReduction_and,int,1,"%d",
	if (value==0) 
		ret=0;
)

//Compute the logical OR of the integers passed by each element.
// The resulting integer will be 1 if any source integer is nonzero; else 0.
SIMPLE_REDUCTION(CkReduction_or,int,0,"%d",
	if (value!=0) 
		ret=1;
)

//Compute the sum the doubles passed by each element.
SIMPLE_REDUCTION(CkReduction_sum_double,double,0.0,"%f",
	ret+=value;
)

//Compute the largest double passed by any element.
SIMPLE_REDUCTION(CkReduction_max_double,double,MINDOUBLE,"%f",
	if (ret<value) 
		ret=value;
)

//Compute the smallest double passed by any element.
SIMPLE_REDUCTION(CkReduction_min_double,double,MAXDOUBLE,"%f",
	if (ret>value) 
		ret=value;
)

/////////////////////// non-simple reductions: set ////////////////
/*
This reducer simply appends the data it recieves from each element
(along with some housekeeping data indicating the source element and data size).
The message data is then a list of reduction_set_element structures
terminated by a dummy reduction_set_element with a sourceElement of -1.
*/

//This rounds an integer up to the nearest multiple of 8
static int RED_ALIGN(int x) {return (~7)&((x)+7);}

//This gives the size (in bytes) of a reduction_set_element
static int REDUCTION_SET_SIZE(int dataSize)
{return RED_ALIGN((2*sizeof(int))+dataSize);}

//This returns a pointer to the next reduction_set_element in the list
static CkReduction_set_element *REDUCTION_SET_NEXT(CkReduction_set_element *cur) 
{
	char *next=((char *)cur)+REDUCTION_SET_SIZE(cur->dataSize);
	return (CkReduction_set_element *)next;
}

//Combine the data passed by each element into an list of reduction_set_elements.
// Each element may contribute arbitrary data (with arbitrary length).
ArrayReductionMessage *CkReduction_set(int nMsg,ArrayReductionMessage **msg)
{
	RED_DEB(("/ PE_%d: reduction_set invoked on %d messages\n",CkMyPe(),nMsg));
	//Figure out how big a message we'll need
	int i,retSize=0;
	for (i=0;i<nMsg;i++)
		if (msg[i]->source<0)
		//This message is composite-- it will just be copied over (less terminating -1)
			retSize+=(msg[i]->dataSize-sizeof(int));
		else //This is a message from an element-- it will be wrapped in a reduction_set_element
			retSize+=REDUCTION_SET_SIZE(msg[i]->dataSize);
	retSize+=sizeof(int);//Leave room for terminating -1.

	RED_DEB(("|- composite set reduction message will be %d bytes\n",retSize));
	
	//Allocate a new message
	ArrayReductionMessage *ret=ArrayReductionMessage::buildNew(retSize,NULL);
	
	//Copy the source message data into the return message
	CkReduction_set_element *cur=(CkReduction_set_element *)(ret->data);
	for (i=0;i<nMsg;i++)
		if (msg[i]->source<0)
		{//This message is composite-- just copy it over (less terminating -1)
			int messageBytes=msg[i]->dataSize-sizeof(int);
			RED_DEB(("|\tmsg[%d] is %d bytes from %d sources\n",i,msg[i]->dataSize,msg[i]->getSources()));
			memcpy((void *)cur,(void *)msg[i]->data,messageBytes);
			cur=(CkReduction_set_element *)(((char *)cur)+messageBytes);
		}
		else //This is a message from an element-- wrap it in a reduction_set_element
		{
			RED_DEB(("|\tmsg[%d] is %d bytes from element %d\n",i,msg[i]->dataSize,msg[i]->source));
			cur->sourceElement=msg[i]->source;
			cur->dataSize=msg[i]->dataSize;
			memcpy((void *)cur->data,(void *)msg[i]->data,msg[i]->dataSize);
			cur=REDUCTION_SET_NEXT(cur);
		}
	cur->sourceElement=-1;//Add a terminating -1.
	RED_DEB(("\\ PE_%d: reduction_set finished-- %d messages combined\n",CkMyPe(),nMsg));
	return ret;
}

//Utility routine: get the next reduction_set_element in the list
// if there is one, or return NULL if there are none.
//To get all the elements, just keep feeding this procedure's output back to
// its input until it returns NULL.
CkReduction_set_element *CkReduction_set_element_next(CkReduction_set_element *cur)
{
	CkReduction_set_element *next=REDUCTION_SET_NEXT(cur);
	if (next->sourceElement==-1)
		return NULL;//This is the end of the list
	else
		return next;//This is just another element
}


