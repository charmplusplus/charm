#include <ckio.h>
#include <errno.h>
#include <algorithm>
#include <sys/stat.h>

#if defined(_WIN32)
#include <io.h>

int pwrite(int fd, const void *buf, size_t nbytes, off_t offset)
{
  long ret = _lseek(fd, offset, SEEK_SET);

  if (ret == -1) {
    return(-1);
  }
  return(_write(fd, buf, nbytes));
}
#define NO_UNISTD_NEEDED
#endif

#if defined(__PGIC__)
// PGI compilers define funny feature flags that lead to standard
// headers omitting this prototype
ssize_t pwrite(int fd, const void *buf, size_t count, off_t offset);
#define NO_UNISTD_NEEDED
#endif

#if !defined(NO_UNISTD_NEEDED)
#include <unistd.h>
#endif

namespace Ck { namespace IO {
    Manager::Manager() : nextToken(0) {
      run();
    }

    void Manager::prepareOutput(const char *name, size_t bytes,
				CkCallback ready, CkCallback complete,
				Options opts) {
      thisProxy[0].prepareOutput_central(name, bytes, ready, complete, opts);
    }

    void Manager::write(Token token, const char *data, size_t bytes, size_t offset) {
      Options &opts = files[token].opts;
      while (bytes > 0) {
        size_t stripeIndex = offset / opts.peStripe;
        int peIndex = stripeIndex % opts.activePEs;
        int pe = opts.basePE + peIndex * opts.skipPEs;
        size_t bytesToSend = std::min(bytes, opts.peStripe - offset % opts.peStripe);
	thisProxy[pe].write_forwardData(token, data, bytesToSend, offset);
	data += bytesToSend;
	offset += bytesToSend;
	bytes -= bytesToSend;
      }
    }

    void Manager::write_forwardData(Token token, const char *data, size_t bytes,
				    size_t offset) {
      //files[token].bufferMap[(offset/stripeSize)*stripeSize] is the buffer to which this char should write to.
      CkAssert(offset + bytes <= files[token].bytes);
      // XXX: CkAssert(this is the right processor to receive this data)

      size_t stripeSize = files[token].opts.peStripe;   
      while(bytes > 0)
      {
	size_t stripeOffset = (offset/stripeSize)*stripeSize;
	size_t expectedBufferSize = std::min(files[token].bytes - stripeOffset, stripeSize);
	struct impl::buffer & currentBuffer = files[token].bufferMap[stripeOffset];
	size_t bytesInCurrentStripe = std::min(expectedBufferSize - offset%stripeSize, bytes);

	//check if buffer this element already exists in map. If not, insert and resize buffer to stripe size
	currentBuffer.expect(expectedBufferSize);

	currentBuffer.insertData(data, bytesInCurrentStripe, offset % stripeSize);

	// Ready to flush?
	if(currentBuffer.isFull()) {
	  //initializa params
	  int l = currentBuffer.bytes_filled_so_far;
	  char *d = &(currentBuffer.array[0]);
	  size_t bufferOffset = stripeOffset;
	  //write to file loop
	  while (l > 0) {
	    CmiInt8 ret = pwrite(files[token].fd, d, l, bufferOffset);
	    if (ret < 0) {
	      if (errno == EINTR) {
		continue;
	      } else {
		CkPrintf("Output failed on PE %d: %s", CkMyPe(), strerror(errno));
		CkAbort("Giving up");
	      }
            }
	    l -= ret;
	    d += ret;
	    bufferOffset += ret;
	  }
	  //write complete - remove this element from bufferMap and call dataWritten
	  thisProxy[0].write_dataWritten(token, currentBuffer.bytes_filled_so_far);
	  files[token].bufferMap.erase(stripeOffset);
	}

	bytes -= bytesInCurrentStripe;
	data += bytesInCurrentStripe;
	offset += bytesInCurrentStripe;
      }
    }

    void Manager::write_dataWritten(Token token, size_t bytes) {
      CkAssert(CkMyPe() == 0);

      files[token].total_written += bytes;

      if (files[token].total_written == files[token].bytes)
	files[token].complete.send();
    }

    void Manager::prepareInput(const char *name, CkCallback ready, Options opts) {
      CkAbort("not yet implemented");
    }

    void Manager::read(Token token, void *data, size_t bytes, size_t offset,
		       CkCallback complete) {
      CkAbort("not yet implemented");
    }

    int Manager::openFile(const std::string& name) {
      int fd;
#if defined(_WIN32)
      fd = _open(name.c_str(), _O_WRONLY | _O_CREAT, _S_IREAD | _S_IWRITE);
#else
      fd = open(name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
#endif
      if (-1 == fd)
	CkAbort("Failed to open a file for parallel output");
      return fd;
    }
  }
}

#include "CkIO.def.h"
