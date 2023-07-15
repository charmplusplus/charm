#include "iotest.decl.h"
#include <vector>
#include <fstream>
#define TEST_FILE "readtest.txt"
#define NUM_READERS 10
CProxy_Main mainProxy;


class Main : public CBase_Main {
	Main_SDAG_CODE;
	Ck::IO::File _file; // the file that is going to be opened
	CProxy_Reader readers; // holds the array of readers
	size_t num_reads_done = NUM_READERS; // initialize the number of reads to 10 and count down
	Ck::IO::Session current_session;
public:
	Main(CkArgMsg* msg){
		thisProxy.startReading();
	}
	// returns a buffer of a sequential read so that the parallel read at offset with number of bytes length can be verified
	char* sequentialRead(size_t offset, size_t bytes){
		char* buffer = new char[bytes + 1];
		int pos = 0;
		std::ifstream ifs(TEST_FILE, std::ios::in | std::ios::binary);
		ifs.seekg(offset);

		while(pos < bytes){
			buffer[pos++] = ifs.get();
		}
		buffer[bytes] = 0;
		ifs.close();
		return buffer;
	}

};

// object that is used to enact the parallel reads
struct Reader : public CBase_Reader {

public:
	Reader(Ck::IO::Session session, size_t bytes, size_t offset, CkCallback after_read){
		size_t my_offset = offset + thisIndex * 10; // the offset to read at; note that it is 10 because the test file is 100 bytes and there are 10 readers, so each reader will read 10 bytes
		ckout << "My offset for reader " << thisIndex << " is " << my_offset << endl;
		Ck::IO::read(session, bytes, my_offset, after_read); 
	}

};

#include "iotest.def.h"
#undef TEST_FILE
