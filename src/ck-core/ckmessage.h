#if !defined(CKMESSAGE_H)
#define CKMESSAGE_H

#include <pup.h>

/**
 * CkMessage is the superclass of all Charm++ messages.
 * Typically, a message foo inherits from CMessage_foo, which
 * inherits from CkMessage.  In the internals of Charm++,
 * messages are often represented by bare "void *"s, which is
 * silly and dangerous.
 */
class CkMessage {
	//Don't use these: use CkCopyMsg
	CkMessage(const CkMessage &);
	void operator=(const CkMessage &);
public:
	CkMessage() {}
	void operator delete(void *ptr) { CkFreeMsg(ptr); }

	/* This pup routine only packs the message itself, *not* the
	message header.  Use CkPupMessage instead of calling this directly. */
	void pup(PUP::er &p);

	/// This is used to display message contents in the debugger.
	static void ckDebugPup(PUP::er &p,void *msg);
};
class CMessage_CkMessage {
public:
	static int __idx;
};

/// CkArgMsg is passed to the mainchare's constructor.
class CkArgMsg : public CkMessage {
public:
  int argc;
  char **argv;
};

#endif
