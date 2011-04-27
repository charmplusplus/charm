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
      CkAssert(offset + bytes <= files[token].bytes);
      // XXX: CkAssert(this is the right processor to receive this data)

      while (bytes > 0) {
	ssize_t ret = pwrite(files[token].fd, data, bytes, offset);
	if (ret < 0)
	  if (errno == EINTR)
	    continue;
	  else {
	    CkPrintf("Output failed on PE %d: %s", CkMyPe(), strerror(errno));
	    CkAbort("Giving up");
	  }
	bytes -= ret;
	data += ret;
	offset += ret;
      }
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
