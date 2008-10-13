#ifndef MSAHASHTABLE_H
#define MSAHASHTABLE_H

#include "ParFUM.h"
#include "ParFUM_internals.h"

typedef UniqElemList<Hashnode> Hashtuple;
typedef MSA1D<Hashtuple,DefaultListEntry<Hashtuple,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DHASH;

class MsaHashtable : private MSA1DHASH
{
public:
	class Read; class Add;
	Read& syncToRead(Add&);
	Add&  syncToAdd(Read&);
	Add& getInitialAdd();
	using MSA1DHASH::pup;
	using MSA1DHASH::enroll;

	class Read : private MSA1DHASH::Read
	{
	public:
		using MSA1DHASH::Read::get;
		friend Read &MsaHashtable::syncToRead(Add&);
		friend Add& MsaHashtable::syncToAdd(Read&);
		void print();

	private:
	Read(MsaHashtable &m) : MSA1DHASH::Read(m) { }
	};

	class Add : private MSA1DHASH::Accum
	{
		using MSA1DHASH::Accum::accumulate;
		friend Add& MsaHashtable::syncToAdd(Read&);
		friend Read &MsaHashtable::syncToRead(Add&);
		friend Add& MsaHashtable::getInitialAdd();
	Add(MsaHashtable &m) : MSA1DHASH::Accum(m) { }
	public:
		int addTuple(int *tuple, int nodesPerTuple, int chunk, int elementNo);

	};


MsaHashtable(int _numSlots,int numWorkers)
	: MSA1DHASH(_numSlots, numWorkers) { }
	MsaHashtable(){};
};



#endif // MSAHASHTABLE_H
