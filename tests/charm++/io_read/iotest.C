#include "iotest.decl.h"
#include <vector>
#include <fstream>

CProxy_Main mainProxy;


class Main : public CBase_Main {
	Main_SDAG_CODE;
	Ck::IO::File _file; // the file that is going to be opened
	CProxy_Reader readers; // holds the array of readers
	size_t num_reads_done = 5; // initialize the number of reads to 5 and count down
	Ck::IO::Session current_session;
public:
	Main(CkArgMsg* msg){
		thisProxy.startReading();
	}

	char* sequentialRead(size_t offset, size_t bytes){
		char* buffer = new char[bytes + 1];
		int pos = 0;
		std::ifstream ifs("readtest.txt");
		ifs.seekg(offset);

		while(pos < bytes){
			buffer[pos++] = ifs.get();
		}
		buffer[bytes] = 0;
		ifs.close();
		return buffer;
	}

};


struct Reader : public CBase_Reader {

public:
	Reader(Ck::IO::Session session, size_t bytes, size_t offset, CkCallback after_read){
		size_t my_offset = offset + thisIndex * 5;
		ckout << "My offset for reader " << thisIndex << " is " << my_offset << endl;
		Ck::IO::read(session, bytes, my_offset, after_read); 
	}

};

#include "iotest.def.h"
