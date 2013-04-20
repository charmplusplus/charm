
#include "seek_block.h"

main::main(CkArgMsg *m)
{
  //normal object construction
  C exampleObject;
  exampleObject.a.x = 0;

  //normal chare array construction
  CProxy_SimpleArray simpleProxy = CProxy_SimpleArray::ckNew(30);

  //pass object to remote method invocation on the chare array
  simpleProxy[29].acceptData(exampleObject);
}

void A::pup(PUP::er& p)
{
	p | x;
	p | y;
	p | z;
}

void B::pup(PUP::er& p)
{
	p | c;
}

void C::pup(PUP::er& p)
{
	// Create two blocks. a is packed in block 0 and b in block 1, but
	// we pup b first on unpack and last on pack.
	PUP::seekBlock s(p,2);
	if (p.isUnpacking()) {
		// pup b first when unpacking
		s.seek(1);
		p | b;
	}

	s.seek(0);
	p | a;

	if (!p.isUnpacking()) {
		// pup b last when packing
		s.seek(1);
		b.pup(p);
	}
	s.endBlock();
}

#include "seek_block.def.h"
