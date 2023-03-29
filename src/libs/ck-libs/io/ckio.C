#include <string>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>

typedef int FileToken;
#include "CkIO.decl.h"
#include "CkIO_impl.decl.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pup_stl.h>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "fs_parameters.h"

using std::min;
using std::max;
using std::map;
using std::string;

/*bool operator==(const Ck::IO::Session& s1, const Ck::IO::Session& s2) {
	return (s1.getFile() == s2.getFile()) && (s1.getBytes() == s2.getBytes()) && (s1.getOffset() == s2.getOffset()) && (s1.getSessionID() == s2.getSessionID());
}
*/


// FROM STACKEXCHANGE: https://stackoverflow.com/questions/19195183/how-to-properly-hash-the-custom-struct
template <class T>
inline void hash_combine(std::size_t & s, const T& v)
{
  std::hash<T> h;
  s^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2);
}

// HASH FOR SESSION
template<>
struct std::hash<Ck::IO::Session> {
	size_t operator()(const Ck::IO::Session& s) const{
		size_t res = 0;
		hash_combine(res, s.getFile());
		hash_combine(res, s.getBytes());
		hash_combine(res, s.getOffset());
		return res;
	}
};


// #define DEBUG 
namespace Ck { namespace IO {
    namespace impl {
      CProxy_Director director;
      CkpvDeclare(Manager *, manager);
    }


    namespace impl {
      struct FileInfo {
        string name;
        CkCallback opened;
        Options opts;
        int fd;
        int sessionID;
        CProxy_WriteSession session;
	CProxy_ReadSession read_session;
        CkCallback complete; // used for the write session complete callback?

        FileInfo(string name_, CkCallback opened_, Options opts_)
          : name(name_), opened(opened_), opts(opts_), fd(-1)
        { }
        FileInfo(string name_, Options opts_)
          : name(name_), opened(), opts(opts_), fd(-1)
        { }
        FileInfo()
          : fd(-1)
        { }
      };

      void fatalError(string desc, string file) {
        CkAbort("FATAL ERROR on PE %d working on file '%s': %s; system reported %s\n",
			CkMyPe(), file.c_str(), desc.c_str(), strerror(errno));
      }

      class Director : public CBase_Director {
        int filesOpened;
        map<FileToken, impl::FileInfo> files;
        CProxy_Manager managers;
        int opnum, sessionID;
        Director_SDAG_CODE

      public:
        Director(CkArgMsg *m)
          : filesOpened(0), opnum(0), sessionID(0)
        {
          delete m;
          director = thisProxy;
          managers = CProxy_Manager::ckNew();
        }

        Director(CkMigrateMessage *m) : CBase_Director(m) { }

        void pup(PUP::er &p) {
          // FIXME: All files must be closed across checkpoint/restart
          if (files.size() != 0)
            CkAbort("CkIO: All files must be closed across checkpoint/restart");

          p | filesOpened;
          p | managers;
          p | opnum;
          p | sessionID;
        }

        void openFile(string name, CkCallback opened, Options opts) {
          if (0 == opts.writeStripe)
            opts.writeStripe = CkGetFileStripeSize(name.c_str());
          if (0 == opts.peStripe)
            opts.peStripe = 4 * opts.writeStripe;
          if (-1 == opts.activePEs)
            opts.activePEs = min(CkNumPes(), 32);
          if (-1 == opts.basePE)
            opts.basePE = 0;
          if (-1 == opts.skipPEs)
            opts.skipPEs = CkMyNodeSize();

          files[filesOpened] = FileInfo(name, opened, opts);
          managers.openFile(opnum++, filesOpened++, name, opts);
        }

        void fileOpened(FileToken file) {
          files[file].opened.send(new FileReadyMsg(file));
        }
	
	// method called by the closeReadSession function from user	
	void closeReadSession(Session read_session, CkCallback after_end){
		CProxy_ReadSession(read_session.sessionID).ckDestroy();
		after_end.send(CkReductionMsg::buildNew(0, NULL, CkReduction::nop)); // invoke a callback

	}

        void prepareWriteSession_helper(FileToken file, size_t bytes, size_t offset,
                                        CkCallback ready, CkCallback complete) {
          Options &opts = files[file].opts;
          files[file].sessionID = sessionID;

	  int numStripes = 0;
	  size_t bytesLeft = bytes, delta = opts.peStripe - offset % opts.peStripe;
	  // Align to stripe boundary
	  if (offset % opts.peStripe != 0 && delta < bytesLeft) {
	    bytesLeft -= delta;
	    numStripes++;
	  }
	  numStripes += bytesLeft / opts.peStripe;
	  if (bytesLeft % opts.peStripe != 0)
	    numStripes++;

          CkArrayOptions sessionOpts(numStripes);
          sessionOpts.setStaticInsertion(true);

          CkCallback sessionInitDone(CkIndex_Director::sessionReady(NULL), thisProxy);
          sessionInitDone.setRefnum(sessionID);
          sessionOpts.setInitCallback(sessionInitDone);

          //sessionOpts.setMap(managers);
          files[file].session =
            CProxy_WriteSession::ckNew(file, offset, bytes, sessionOpts);
          CkAssert(files[file].complete.isInvalid());
          files[file].complete = complete;
        }
	
// // 	// called by user-facing read call to facilitate the actual read
// 	void read(Session session, size_t bytes, size_t offset, CkCallback after_read){
// 		CProxy_ReadAssembler ra = CProxy_ReadAssembler::ckNew(session, bytes, offset, after_read); // create read assembler
// 		Options& opt = files[session.file].opts;	
// 		size_t read_stripe = opt.read_stripe;
// 		size_t start_idx = offset / read_stripe; // the first index that has the relevant data
// 		// CkPrintf("Read request of %d bytes starting at %d\n", bytes, offset);
// 		for(size_t i = start_idx; (i * read_stripe) < (offset + bytes); ++i){
// 			// tell all the chares that have data to search and send
// 			CProxy_ReadSession(session.sessionID)[i].sendData(offset, bytes, ra); 
// 		}
// 	}
// 
// 	// called by user-facing read call to facilitate the actual read
// 	void read(Session session, size_t bytes, size_t offset, CkCallback after_read, size_t tag){
// 		CProxy_ReadAssembler ra = CProxy_ReadAssembler::ckNew(session, bytes, offset, after_read, tag); // create read assembler
// 		Options& opt = files[session.file].opts;	
// 		size_t read_stripe = opt.read_stripe;
// 		size_t start_idx = offset / read_stripe; // the first index that has the relevant data
// 		// CkPrintf("Read request of %d bytes starting at %d\n", bytes, offset);
// 		for(size_t i = start_idx; (i * read_stripe) < (offset + bytes); ++i){
// 			// tell all the chares that have data to search and send
// 			CProxy_ReadSession(session.sessionID)[i].sendData(offset, bytes, ra); 
// 		}
// 	}

		
	void prepareReadSessionHelper(FileToken file, size_t bytes, size_t offset, CkCallback ready){
		if(!bytes){
			CkPrintf("You're tryna read 0 bytes buddy\n");
			CkAbort("You're tryna read 0 bytes. Oops.\n");
		}
		size_t session_bytes = bytes; // amount of bytes in the session
		// ckout << "In prepare read session helper" << endl;
		Options& opts = files[file].opts;
		files[file].sessionID = sessionID;
		// determine the number of reader sessions required, depending on the session size and the number of bytes per reader
		int num_readers = 0;
		size_t remainder = bytes % opts.read_stripe;
		if(remainder){
			num_readers++;
		}
		num_readers += (bytes / opts.read_stripe); 
		CkArrayOptions sessionOpts(num_readers); // set the number of elements in the chare array
		CkCallback sessionInitDone(CkIndex_Director::sessionReady(0), thisProxy);
		sessionInitDone.setRefnum(sessionID);
		sessionOpts.setInitCallback(sessionInitDone); // invoke the sessionInitDone callback after all the elements of the chare array are created
		files[file].read_session = CProxy_ReadSession::ckNew(file, offset, bytes, sessionOpts); // create the readers
	}

        void sessionComplete(FileToken token) {
          CProxy_CkArray(files[token].session.ckGetArrayID()).ckDestroy();
          files[token].complete.send(CkReductionMsg::buildNew(0, NULL, CkReduction::nop));
          files[token].complete = CkCallback(CkCallback::invalid);
        }

        void close(FileToken token, CkCallback closed) {
          managers.close(opnum++, token, closed);
          files.erase(token);
        }
      };

	 // Stores information about a particular read
      struct ReadInfo {
		size_t bytes_left; // the number of bytes the user wants for a particular read
		size_t read_bytes;
		size_t read_offset; // the offset they specify for their read
		size_t read_size;
		CkCallback after_read; // the callback to invoke after the read is complete
		ReadCompleteMsg* msg;
		size_t read_tag = -1;
      };
    // class that is used to aggregate the data for a specific read call made by the user
    class ReadAssembler : public CBase_ReadAssembler {
	private:
		Session _session;
		std::unordered_map<int, ReadInfo> _read_info_buffer; // matches an assigned tag for a read to the read info
		size_t _curr_read_tag = 0;
		size_t _curr_RDMA_tag = 0;
	public:
		ReadAssembler(Session session){
			_session = session;
		}
	
		size_t addReadToTable(size_t read_bytes, size_t read_offset, CkCallback after_read){
			// do the initialization of the read struct
			ReadInfo ri;
			ri.bytes_left = read_bytes;
			ri.read_offset = read_offset;
			ri.read_bytes = read_bytes;
			ri.after_read = after_read;
			ri.msg = new(read_bytes) ReadCompleteMsg();
			ri.msg -> offset = read_offset;
			ri.msg -> bytes = read_bytes;
			if(_read_info_buffer.count(_curr_read_tag) != 0){
				CkPrintf("Something is wrong, a read tag is being overwritten on pe=%d!\n", CkMyPe());
				CkExit();
			}
			_read_info_buffer[_curr_read_tag] = ri; // put the readinfo struct in the table
			// CkPrintf("On PE=%d, just allocated read_tag=%zu for new read\n", CkMyPe(), _curr_tag);
			_curr_read_tag++;
			// CkPrintf("New tag: %d. Returning %d\n", _curr_tag, _curr_tag - 1);
			return _curr_read_tag - 1;
		}
		
		void removeEntryFromReadTable(int tag) {
			_read_info_buffer.erase(tag);
		}	

		void shareData(int read_tag, int buffer_tag, size_t read_chare_offset, size_t num_bytes, char* data, CkNcpyBufferPost* ncpyPost){
			// CkPrintf("About to match buffer for read_tag=%d, buffer_tag=%d, pe=%d\n", read_tag, buffer_tag, CkMyPe());
			ncpyPost[0].regMode = CK_BUFFER_REG;
			ncpyPost[0].deregMode = CK_BUFFER_DEREG;
			CkMatchBuffer(ncpyPost, 0, buffer_tag);
			// CkPrintf("Successfully matched buffer for read_tag=%d, buffer_tag=%d, pe=%d\n", read_tag, buffer_tag, CkMyPe());
		}
		
		void shareData(int read_tag, int buffer_tag, size_t read_chare_offset, size_t num_bytes, char* data){
			// size_t start_idx = read_chare_offset - _read_offset; // start index for writing to _data_buffer
			// copy over the data from data to the correct place in the _data_buffer
//			for(size_t counter = 0; counter < num_bytes; ++counter){
			// 	char ch = data[counter];
			// 	_data_buffer[start_idx + counter] = ch;
			// }
			#ifdef DEBUG
				CkPrintf("shareData args: read_chare_offset=%zu, num_bytes=%zu, data=%p. start_idx = %zu\n", read_chare_offset, num_bytes, data, start_idx);
			#endif
			ReadInfo info = _read_info_buffer[read_tag]; // get the struct from the buffer tag
			// memcpy(msg -> data + start_idx, data, num_bytes); // copying the num_bytes from data to the read buffer
			info.bytes_left -= num_bytes; // decrement the number of remaining bytes to read
			if(info.bytes_left) return; // if there are bytes still to read, just return
			// CkPrintf("Read with read_tag=%d, PE=%d is complete\n", read_tag, CkMyPe());
			info.after_read.send(info.msg);
			removeEntryFromReadTable(read_tag); // the read is complete; remove it from the table
		}

		void serveRead(size_t read_bytes, size_t read_offset, CkCallback after_read, size_t read_stripe){
			CkPrintf("In serveRead on PE=%d\n", CkMyPe());
			int read_tag = addReadToTable(read_bytes, read_offset, after_read); // create a tag for the actual read
			// CkPrintf("Got a new read_tag=%zu for read; PE=%d\n", read_tag, CkMyPe());
			// get the necessary info
			size_t bytes = read_bytes;
			size_t start_idx = (read_offset - _session.offset)/ read_stripe; // the first index that has the relevant data
			ReadInfo& info = _read_info_buffer[read_tag]; // get the ReadInfo object
			ReadCompleteMsg* msg = info.msg;
			// CkPrintf("Read request of %d bytes starting at %d\n", bytes, offset);
			for(size_t i = start_idx; (i * read_stripe) < (read_offset + bytes); ++i){
				size_t data_idx;
				size_t data_len;
				if(i == start_idx){
					data_idx = 0; // at the start of read
					// if intrabuffer, just take read size; o/w go from offset to end of buffer chare
					data_len = std::min((read_stripe * (i+1) + _session.offset - read_offset), bytes);
				} else {
					data_idx = (read_stripe * i + _session.offset - read_offset); // first byte of fille in buffer chare, offset from the read offset
					data_len = std::min(read_stripe, read_offset + bytes - (read_stripe * i + _session.offset));
					// the length is gonna be the entire chare's readstripe, or what's remainig of the read
				}
				// do the CkPost call
				// int tag = _curr_RDMA_tag++;
				int tag = getRDMATag(); 
				// CkPrintf("About to post buffer with read_tag=%d, 0_copy_tag=%d on pe=%d\n", read_tag, tag, CkMyPe());
				CkPostBuffer(msg -> data + data_idx, data_len, tag);
				CProxy_ReadSession(_session.sessionID)[i].sendData(read_tag, tag, read_offset, bytes, thisProxy, CkMyPe()); 
			}

		}
    	};



      class Manager : public CBase_Manager {
        Manager_SDAG_CODE
        int opnum;
	std::unordered_map<Session, CProxy_ReadAssembler> _session_to_read_assembler; // map used to 
	// get the read assembler for a specific session
	int _curr_tag = 0;
      public:
        Manager()
          : opnum(0)
        {
          CkpvInitialize(Manager*, manager);
          CkpvAccess(manager) = this;
          thisProxy[CkMyPe()].run();
        }

        Manager(CkMigrateMessage *m)
          : CBase_Manager(m)
        {
          CkpvInitialize(Manager*, manager);
          CkpvAccess(manager) = this;
        }
	
	// invoked to insert the readassembler for a specific session
	void addSessionReadAssemblerMapping(Session session, CProxy_ReadAssembler ra, CkCallback ready){
		_session_to_read_assembler[session] = ra;
		CkPrintf("ON PE=%d, is the session in the buffer: %d\n", CkMyPe(), _session_to_read_assembler.count(session));
		CkCallback cb(CkIndex_Director::addSessionReadAssemblerFinished(0), director);
		int temp = 0;
		contribute(sizeof(temp), &temp, CkReduction::nop, cb); // effectively a barrier, makes sure every PE is done with adding session
	}
		
	int getTag(){
		return _curr_tag++;
	}

        void pup(PUP::er &p) {
          p | opnum;

          // FIXME: All files must be closed across checkpoint/restart
          if (files.size() != 0)
            CkAbort("CkIO: All files must be closed across checkpoint/restart");
        }

        void prepareFile(FileToken token, string name, Options opts) {
          CkAssert(files.end() == files.find(token));
          //CkAssert(lastActivePE(opts) < CkNumPes());
          CkAssert(opts.writeStripe <= opts.peStripe);
          files[token] = impl::FileInfo(name, opts);

          contribute(sizeof(FileToken), &token, CkReduction::max_int,
                     CkCallback(CkReductionTarget(Director, fileOpened), director));
        }

        impl::FileInfo* get(FileToken token) {
          CkAssert(files.find(token) != files.end());

          // Open file if we're one of the active PEs
          // XXX: Or maybe wait until the first write-out, to smooth the metadata load?
          if (files[token].fd == -1) {
            string& name = files[token].name;
#if defined(_WIN32)
            int fd = CmiOpen(name.c_str(), _O_WRONLY | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
            int fd = CmiOpen(name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
#endif
            if (-1 == fd)
              fatalError("Failed to open a file for parallel output", name);

            files[token].fd = fd;
          }

          return &(files[token]);
        }
	
	void read(Session session, size_t bytes, size_t offset, CkCallback after_read){
		CkPrintf("manager %d just got a read request of size=%zu, offest=%zu\n", CkMyPe(), bytes, offset);
		if(!_session_to_read_assembler.count(session)){
			CkPrintf("Why is there no session associated with read on manager %d\n", CkMyPe());
			CkExit();
		}
		CProxy_ReadAssembler ra = _session_to_read_assembler[session];	
		Options& opt = files[session.file].opts;	
		size_t read_stripe = opt.read_stripe;
		// CProxy_ReadAssembler(ra)[CkMyPe()].serveRead(bytes, offset, after_read,read_stripe); // actually does grunt-work of serving data
	 	ReadAssembler* grp_ptr = ra.ckLocalBranch();
		if(!grp_ptr){
			CkPrintf("The pointer to the local branch is null on pe=%d\n", CkMyPe());
			CkExit();
		}
	 	grp_ptr -> serveRead(bytes, offset, after_read, read_stripe);
	}

        void write(Session session, const char *data, size_t bytes, size_t offset) {
          Options &opts = files[session.file].opts;
          size_t stripe = opts.peStripe;

          CkAssert(offset >= session.offset);
          CkAssert(offset + bytes <= session.offset + session.bytes);

          size_t sessionStripeBase = (session.offset / stripe) * stripe;

          while (bytes > 0) {
            size_t stripeIndex = (offset - sessionStripeBase) / stripe;
            size_t bytesToSend = min(bytes, stripe - offset % stripe);

            CProxy_WriteSession(session.sessionID)[stripeIndex]
              .forwardData(data, bytesToSend, offset);

            data += bytesToSend;
            offset += bytesToSend;
            bytes -= bytesToSend;
          }
        }

        void doClose(FileToken token, CkCallback closed) {
          int fd = files[token].fd;
          if (fd != -1) {
            int ret;
            do {
#if defined(_WIN32)
              ret = _close(fd);
#else
              ret = ::close(fd);
#endif
            } while (ret < 0 && errno == EINTR);
            if (ret < 0)
              fatalError("close failed", files[token].name);
          }
          files.erase(token);
          contribute(closed);
        }

        int procNum(int arrayHdl,const CkArrayIndex &element)
        {
#if 0
          int peIndex = stripeIndex % opts.activePEs;
          int pe = opts.basePE + peIndex * opts.skipPEs;
#endif
          return 0;
        }

      private:
        map<FileToken, impl::FileInfo> files;

        int lastActivePE(const Options &opts) {
          return opts.basePE + (opts.activePEs-1)*opts.skipPEs;
        }
      };
	
      int getRDMATag() { // function to allow ReadAssembler to get the current RDMA tag on the PE
	return CkpvAccess(manager) -> getTag();
      }

      class WriteSession : public CBase_WriteSession {
        const FileInfo *file;
        size_t sessionOffset, myOffset;
        size_t sessionBytes, myBytes, myBytesWritten;
        FileToken token;

        struct buffer {
          std::vector<char> array;
          int bytes_filled_so_far;

          buffer() {
            bytes_filled_so_far = 0;
          }

          void expect(size_t bytes) {
            array.resize(bytes);
          }

          void insertData(const char *data, size_t length, size_t offset) {
            char *dest = &array[offset];
            memcpy(dest, data, length);

            bytes_filled_so_far += length;
          }

          bool isFull() {
            return bytes_filled_so_far == array.size();
          }
        };
        map<size_t, struct buffer> bufferMap;

      public:
        WriteSession(FileToken file_, size_t offset_, size_t bytes_)
          : file(CkpvAccess(manager)->get(file_))
          , sessionOffset(offset_)
          , myOffset((sessionOffset / file->opts.peStripe + thisIndex)
                     * file->opts.peStripe)
          , sessionBytes(bytes_)
          , myBytes(min(file->opts.peStripe, sessionOffset + sessionBytes - myOffset))
          , myBytesWritten(0)
          , token(file_)
        {
          CkAssert(file->fd != -1);
          CkAssert(myOffset >= sessionOffset);
          CkAssert(myOffset + myBytes <= sessionOffset + sessionBytes);
        }

        WriteSession(CkMigrateMessage *m) { }

        void forwardData(const char *data, size_t bytes, size_t offset) {
          CkAssert(offset >= myOffset);
          CkAssert(offset + bytes <= myOffset + myBytes);

          size_t stripeSize = file->opts.writeStripe;

          while (bytes > 0) {
            size_t stripeBase = (offset/stripeSize)*stripeSize;
            size_t stripeOffset = max(stripeBase, myOffset);
            size_t nextStripe = stripeBase + stripeSize;
            size_t expectedBufferSize = min(nextStripe, myOffset + myBytes) - stripeOffset;
            size_t bytesInCurrentStripe = min(nextStripe - offset, bytes);

            buffer& currentBuffer = bufferMap[stripeOffset];
            currentBuffer.expect(expectedBufferSize);

            currentBuffer.insertData(data, bytesInCurrentStripe, offset - stripeOffset);

            if (currentBuffer.isFull()) {
              flushBuffer(currentBuffer, stripeOffset);
              bufferMap.erase(stripeOffset);
            }

            bytes -= bytesInCurrentStripe;
            data += bytesInCurrentStripe;
            offset += bytesInCurrentStripe;
          }

          if (myBytesWritten == myBytes)
            contribute(CkCallback(CkIndex_WriteSession::syncData(), thisProxy));
        }

        void syncData() {
          int status;
          CkAssert(bufferMap.size() == 0);
#if CMK_HAS_FDATASYNC_FUNC
          while ((status = fdatasync(file->fd)) < 0) {
            if (errno != EINTR) {
              fatalError("fdatasync failed", file->name);
            }
          }
#elif CMK_HAS_FSYNC_FUNC
          while ((status = fsync(file->fd)) < 0) {
            if (errno != EINTR) {
              fatalError("fsync failed", file->name);
            }
          }
#elif defined(_WIN32)
          intptr_t hFile = _get_osfhandle(file->fd);
          if (FlushFileBuffers((HANDLE)hFile) == 0)
            fatalError("FlushFileBuffers failed", file->name);
#elif CMK_HAS_SYNC_FUNC
#warning "Will call sync() for every completed write"
          sync(); // No error reporting from sync()
#else
#warning "No file synchronization function available!"
#endif

          contribute(sizeof(FileToken), &token, CkReduction::max_int,
                     CkCallback(CkReductionTarget(Director, sessionComplete), director));
        }

        void flushBuffer(buffer& buf, size_t bufferOffset) {
          int l = buf.bytes_filled_so_far;
          char *d = &(buf.array[0]);

          CmiInt8 ret = CmiPwrite(file->fd, d, l, bufferOffset);
          if (ret < 0)
            fatalError("Call to pwrite failed", file->name);

          CkAssert(ret == l);
          myBytesWritten += l;
        }
      };

	      

      class ReadSession : public CBase_ReadSession {
	private:
		FileToken _token; // the token of the given file
      		const FileInfo* _file; // the pointer to the FileInfo
		size_t _session_bytes; // number of bytes in the session
		size_t _session_offset; // the offset of the session
		size_t _my_offset;
		size_t _my_bytes;
		std::vector<char> _buffer;
		
	public:
		ReadSession(FileToken file, size_t offset, size_t bytes) : _token(file), _file(CkpvAccess(manager)->get(file)), _session_bytes(bytes), _session_offset(offset){
			_my_offset = thisIndex * (_file -> opts.read_stripe) + _session_offset;
			_my_bytes = min(_file -> opts.read_stripe, _session_offset + _session_bytes - _my_offset); // get the number of bytes owned by the session
			CkAssert(_file -> fd != -1);
			CkAssert(_my_offset >= _session_offset);
			CkAssert(_my_offset + _my_bytes <= _session_offset + _session_bytes);
			double disk_read_before = CkWallTimer(); // get the before disk_read
			readData();
			double disk_read_after = CkWallTimer(); // end disk time
			double total_time = disk_read_after - disk_read_before;
			CkCallback cb(CkReductionTarget(ReadSession, printTime), thisProxy[0]);
			contribute(sizeof(double), &total_time, CkReduction::max_double, cb);
			
		}

		void clearBuffer() {
			_buffer.clear(); // clears the buffer
		}

		void printTime(double time_taken){
			CkPrintf("The time to disk took %f seconds\n", time_taken);
		
		}


		void readData(){
		//	std::ifstream ifs(_file -> name); // open the file
		//	if(ifs.fail()){ // error handling if opening the file failed
		//		CkAbort("Couldn't open the file %s\n", (_file -> name).c_str());
		//	}
		//	ifs.seekg(_my_offset); // jump to the point where the chare should start reading
			FILE* fp = fopen(_file -> name.c_str(), "rb"); // open the file pointer
			if(!fp){
				CkPrintf("Opening of the file %s went wrong\n", _file -> name.c_str());
			}
			_buffer.resize(_my_bytes, 'z'); // resize it and init with 'z' to denote what hasn't been changed
			_buffer.shrink_to_fit(); // get rid of any extra capacity 
			char* buffer = _buffer.data(); // point to the underlying char* of the vector; does not own the array
			fseek(fp, _my_offset, SEEK_SET);
			size_t num_bytes_read = fread(buffer, 1, _my_bytes, fp);
			if(!num_bytes_read){
				if(ferror(fp))
					CkPrintf("Something went wrong trying to open the file %s\n", _file -> name.c_str());
				else if (feof(fp))
					CkPrintf("Reached the EOF\n");
			} else if (num_bytes_read != _my_bytes) {
				CkPrintf("Supposed to read %zu bytes, but only read %zu bytes\n", _my_bytes, num_bytes_read);
			}
			fclose(fp);
			
		}	
		
		// the method by which you send your data to the ra chare
		void sendData(int read_tag, int buffer_tag, size_t offset, size_t bytes, CProxy_ReadAssembler ra, int pe){
			// CkPrintf("sending data for read)tag=%d and buffer_tag=%d from ReadSession[%d]\n", read_tag, buffer_tag, thisIndex);
			size_t chare_offset;
			size_t chare_bytes;

			if (offset >= (_my_offset + _my_bytes)) return; // read call starts to the right of this chare
			
			else if(offset + bytes <= _my_offset) return; // the read call starts to the left of this chare


			if(offset < _my_offset) chare_offset = _my_offset; // the start of the read is below this chare, so we should read in the current data from start of what it owns
			else chare_offset = offset; // read offset is in the middle

			size_t end_byte_chare = min(offset + bytes, _my_offset + _my_bytes); // the last byte, exclusive, this chare should read
			size_t bytes_to_read = end_byte_chare - chare_offset;
			// std::vector<char> data_to_send; 
			// data_to_send.resize(bytes_to_read); // the number of bytes to read to avoid the pushback function
			// data_to_send.shrink_to_fit(); // will this improve it? hopefully sending less data over network is good
			// // remoe copy, just send poitner to start of data needed
			// while(bytes_read < bytes_to_read){ // still bytes to read and haven't gone out of bounds
			// 	char data = _buffer[chare_offset - _my_offset + bytes_read]; // get the data we want
			// 	data_to_send.at(bytes_read) = data;
			// 	bytes_read++;
			// }
			#ifdef DEBUG
				CkPrintf("chare_offset=%zu, end_byte_chare=%zu, bytes_to_read=%zu, offset=%zu, bytes=%zu\n", chare_offset, end_byte_chare, bytes_to_read, offset, bytes);	
			#endif
			CkCallback cb(CkIndex_ReadSession::zeroCopyCallback(chare_offset, CkWallTimer()), thisProxy[thisIndex]);	
			CProxy_ReadAssembler(ra)[pe].shareData(read_tag, buffer_tag, chare_offset, bytes_to_read, CkSendBuffer(_buffer.data() + (chare_offset - _my_offset)/*, cb*/)); // send this data to the ReadAssembler
		}

		void zeroCopyCallback(size_t offset, double time){
			double end = CkWallTimer();
			double diff = end - time;
			CkPrintf("Zero copy with offset at %zu completed in %f seconds\n", offset, diff);
		}
      };

      class Map : public CBase_Map {
      public:
        Map()
          { }

        int procNum(int arrayHdl, const CkArrayIndex &element) {
          return 0;
        }
      };

    } // END OF IMPL

    void open(string name, CkCallback opened, Options opts) {
      impl::director.openFile(name, opened, opts);
    }

    void startSession(File file, size_t bytes, size_t offset,
                      CkCallback ready, CkCallback complete) {
      impl::director.prepareWriteSession(file.token, bytes, offset, ready, complete);
    } 

    void startReadSession(File file, size_t bytes, size_t offset, CkCallback ready){
	#ifdef DEBUG
		CkPrintf("This is DEBUG for the reads\n");
	#endif
	impl::director.prepareReadSession(file.token, bytes, offset, ready);
    }
    

    void closeReadSession(Session read_session, CkCallback after_end){
	impl::director.closeReadSession(read_session, after_end); // call the director helper
    }


    void startSession(File file, size_t bytes, size_t offset, CkCallback ready,
                      const char *commitData, size_t commitBytes, size_t commitOffset,
                      CkCallback complete) {
      impl::director.prepareWriteSession(file.token, bytes, offset, ready,
                                         commitData, commitBytes, commitOffset,
                                         complete);
    }

    void write(Session session, const char *data, size_t bytes, size_t offset) {
        using namespace impl;
        CkpvAccess(manager)->write(session, data, bytes, offset);
    }

    void read(Session session, size_t bytes, size_t offset, CkCallback after_read){
	CkAssert(bytes <= session.bytes);
	CkAssert(offset + bytes <= session.offset + session.bytes);
	// call the director function to facilitate the actual read
	// impl::director.read(session, bytes, offset, after_read);
	 using namespace impl;
	 CkpvAccess(manager) -> read(session, bytes, offset, after_read);
    }      

   //  void read(Session session, size_t bytes, size_t offset, CkCallback after_read, size_t tag){
   //      CkAssert(bytes <= session.bytes);
   //      CkAssert(offset + bytes <= session.offset + session.bytes);
   //      // call the director function to facilitate the actual read
   //      impl::director.read(session, bytes, offset, after_read, tag);
   //  }      

    void close(File file, CkCallback closed) {
      impl::director.close(file.token, closed);
    }


    class SessionCommitMsg : public CMessage_SessionCommitMsg {

    };

  }
}

#include "CkIO.def.h"
#include "CkIO_impl.def.h"
