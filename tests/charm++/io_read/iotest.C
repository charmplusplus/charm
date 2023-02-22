#include "iotest.decl.h"
#include <vector>
#include <fstream>
#define TEST_FILE "readtest.txt"
CProxy_Main mainProxy;
using std::min;

class Main : public CBase_Main {
	Main_SDAG_CODE;
	Ck::IO::File _file; // the file that is going to be opened
	CProxy_Reader readers; // holds the array of readers
	size_t num_reads_done = 5; // initialize the number of reads to 5 and count down
	size_t file_size = 0;
	Ck::IO::Session current_session;
public:
	Main(CkArgMsg* msg){
		std::ifstream ifs(TEST_FILE);
		ifs.seekg(0, ifs.end);
		file_size = ifs.tellg();
		ifs.close();
		thisProxy.startReading();
	}
	// returns a buffer of a sequential read so that the parallel read at offset with number of bytes length can be verified
	char* sequentialRead(size_t offset, size_t bytes){
		char* buffer = new char[bytes + 1];
		int pos = 0;
		std::ifstream ifs(TEST_FILE);
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
	Reader(Ck::IO::Session session, size_t bytes, size_t offset, size_t file_size, CkCallback after_read){
		size_t my_offset = offset + thisIndex * (file_size / 5);
		
		size_t end_byte = my_offset + (file_size / 5);
		if(thisIndex == 4) end_byte = bytes;
		size_t num_bytes_to_read = end_byte - my_offset;
		// CkPrintf("For reader %d, the arguments were %zu, %zu, %zu)\n", thisIndex, bytes, offset, file_size);
		// CkPrintf("My offset for reader %zu is %zu and I will be reading %zu bytes\n", thisIndex, my_offset, num_bytes_to_read);
		// Ck::IO::read(session, num_bytes_to_read, my_offset, after_read); 
		Ck::IO::read(session, 8742, 37 + thisIndex * 8742, after_read);
	}

};

#include "iotest.def.h"
#undef TEST_FILE
