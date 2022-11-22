#include "iotest.decl.h"
#include <vector>

class Main : public CBase_Main {
	Main_SDAG_CODE;
	Ck::IO::File _file; // the file that is going to be opened
public:
	Main(CkArgMsg* msg){
		thisProxy.startReading();
	}



};


struct Reader : public CBase_Reader {

public:
	Reader(Ck::IO::Session session, size_t bytes, size_t offset, CkCallback after_read){
		size_t my_offset = offset + thisIndex * 10;
		Ck::IO::read(session, bytes, my_offset, after_read); 
	}

};
