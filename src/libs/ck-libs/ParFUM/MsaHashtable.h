#ifndef MSAHASHTABLE_H
#define MSAHASHTABLE_H

#include "ParFUM.h"
#include "ParFUM_internals.h"

typedef UniqElemList<Hashnode> Hashtuple;
typedef MSA::MSA1D<Hashtuple,DefaultListEntry<Hashtuple,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DHASH;

class MsaHashtable
{
    MSA1DHASH msa;
    bool initHandleGiven;

public:
	class Read; class Add;
	friend class Read; friend class Add;
	Add getInitialAdd();
	void pup(PUP::er &p) { p|msa; }
	void enroll(int n) { msa.enroll(n); }

	class Read : private MSA::MSARead<MSA1DHASH>
	{
	public:
	    using MSA::MSARead<MSA1DHASH>::get;

		friend class MsaHashtable;
		void print();
		Add syncToAdd();

	private:
	Read(MSA1DHASH *m) : MSA::MSARead<MSA1DHASH>(m) { }
	};

	class Add : private MSA::MSAAccum<MSA1DHASH>
	{
	    using MSA::MSAAccum<MSA1DHASH>::accumulate;
	    friend class MsaHashtable;
	Add(MSA1DHASH *m) : MSA::MSAAccum<MSA1DHASH>(m) { }
	public:
		int addTuple(int *tuple, int nodesPerTuple, int chunk, int elementNo);
		Read syncToRead();
	};


MsaHashtable(int _numSlots,int numWorkers)
    : msa(_numSlots, numWorkers), initHandleGiven(false) { }
	MsaHashtable(){};
};



#endif // MSAHASHTABLE_H
