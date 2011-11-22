#include <ckio.h>
#include <errno.h>
#include <algorithm>


namespace Ck { namespace IO {
    Manager::Manager() : nextToken(0) {
      __sdag_init();
      run();
    }

    void Manager::prepareOutput(const char *name, size_t bytes,
				CkCallback ready, CkCallback complete,
				Options opts) {
      thisProxy[0].prepareOutput_central(name, bytes, ready, complete, opts);
    }

    void Manager::write(Token token, const char *data, size_t bytes, size_t offset) {
      Options &opts = files[token].opts;
      do {
	size_t stripe = offset / opts.peStripe;
	int pe = opts.basePE + stripe * opts.skipPEs;
	size_t bytesToSend = std::min(bytes, opts.peStripe - offset % opts.peStripe);
	thisProxy[pe].write_forwardData(token, data, bytesToSend, offset);
	data += bytesToSend;
	offset += bytesToSend;
	bytes -= bytesToSend;
      } while (bytes > 0);
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
	size_t expectedBufferSize = std::min(files[token].bytes - stripeOffset, stripeSize) ;
	struct buffer & currentBuffer = files[token].bufferMap[stripeOffset];
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
	    ssize_t ret = pwrite(files[token].fd, d, l, bufferOffset);
	    if (ret < 0)
	      if (errno == EINTR)
		continue;
	      else {
		CkPrintf("Output failed on PE %d: %s", CkMyPe(), strerror(errno));
		CkAbort("Giving up");
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
  }
}

#include "CkIO.def.h"
