#include "ParFUM.h"
#include "ParFUM_internals.h"
#include "MsaHashtable.h"

using namespace MSA;

void operator+=(Hashtuple &t, const Hashnode &n)
{
    t.vec->push_back(n);
}

MsaHashtable::Add MsaHashtable::getInitialAdd()
{
	if(initHandleGiven)
		throw MSA_InvalidHandle();
	
	initHandleGiven = true;
	return Add(&this->msa);
}

MsaHashtable::Add MsaHashtable::Read::syncToAdd()
{
    syncDone();
    return Add(msa);
}

MsaHashtable::Read MsaHashtable::Add::syncToRead()
{
	syncDone();
	return Read(msa);
}

void MsaHashtable::Read::print()
{
	char str[100];
	for(int i=0; i < msa->length(); i++) {
		const Hashtuple &t = get(i);
		for(int j=0;j<t.vec->size();j++){
			Hashnode &tuple = (*t.vec)[j];
			printf("ghost element chunk %d element %d index %d tuple < %s>\n", 
			       tuple.chunk, tuple.elementNo, i, 
			       tuple.nodes.toString(tuple.numnodes,str));
		}
	}
}

int MsaHashtable::Add::addTuple(int *tuple,int nodesPerTuple,int chunk,int elementNo)
{
    int slots = msa->length();
	// sort the tuples to get a canonical form
	// bubble sort should do just as well since the number
	// of nodes is less than 10.
	for(int i=0;i<nodesPerTuple-1;i++){
		for(int j=i+1;j<nodesPerTuple;j++){
			if(tuple[j] < tuple[i]){
				int t = tuple[j];
				tuple[j] = tuple[i];
				tuple[i] = t;
			}
		}
	}

	//find out the index
	CmiUInt8 sum = 0;
	for(int i=0;i<nodesPerTuple;i++){
		sum = sum*slots + tuple[i];
	}
	int index = (int )(sum %(long )slots);
	Hashnode entry(nodesPerTuple,chunk,elementNo,tuple);

	accumulate(index) += entry;
	char str[100];
	DEBUG(printf("[%d] adding tuple %s element %d to index %d \n",chunk,entry.nodes.toString(nodesPerTuple,str),elementNo,index));
	return index;
}

