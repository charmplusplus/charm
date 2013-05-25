#ifndef BIGSIM_RECORD_H
#define BIGSIM_RECORD_H

/// Message watcher: for record/replay support
class BgMessageWatcher {
protected:
        int nodelevel;
public:
        virtual ~BgMessageWatcher() {}
        /**
         * This message is about to be processed by Charm.
         * If this function returns false, the message will not be processed.
         */
        virtual bool record(char *msg) { return false; }
        virtual int replay() { return 0; }
	virtual void rewind() {}
};

class BgMessageRecorder : public BgMessageWatcher {
        FILE *f;
	long pos;
public:
        BgMessageRecorder(FILE * f_, int node);
        ~BgMessageRecorder() { fclose(f); }

        virtual bool record(char *msg) {
//                if (BgGetGlobalWorkerThreadID()==0) printf("srcpe: %d size: %d handle: %d\n",CmiBgMsgSrcPe(msg),CmiBgMsgLength(msg),CmiBgMsgHandle(msg));
                int d = CmiBgMsgSrcPe(msg);
		pos = ftell(f);
                fwrite(&d, sizeof(int), 1, f);
//CmiAssert(CmiBgMsgThreadID(msg) != -1);
                if ( (nodelevel == 0 && d == BgGetGlobalWorkerThreadID()) ||
                     (nodelevel == 1 && d/BgGetNumWorkThread() == BgGetGlobalWorkerThreadID()/BgGetNumWorkThread()) ) {
                    //CmiPrintf("[%d] local message.\n", BgGetGlobalWorkerThreadID());
                    return true; // don't record local msg
                }
                d = CmiBgMsgLength(msg);
                fwrite(&d, sizeof(int), 1, f);
/*
if (BgGetGlobalWorkerThreadID()==1 && CmiBgMsgHandle(msg) == 21) {
int *m = (int *) ((char *)msg+CmiReservedHeaderSize);
printf("replay: %d %d\n", m[0], m[1]);
}
*/
                fwrite(msg, sizeof(char), d, f);
                //CmiPrintf("[%d] BgMessageRecord>  PE: %d size: %d msg: %p\n", BgGetGlobalWorkerThreadID(), CmiBgMsgSrcPe(msg),CmiBgMsgLength(msg), msg);
                return true;
        }
        virtual int replay() { return 0; }
	virtual void rewind() {
//if (BgGetGlobalWorkerThreadID()==0) printf("rewind to %ld\n", pos);
		fseek(f, pos, SEEK_SET);
	}
	void write_nodeinfo();
};

class BgMessageReplay : public BgMessageWatcher {
        FILE * f;
        int lcount, rcount;
        /// Read the next message we need from the file:
private:
	void done() {
   		int mype = BgGetGlobalWorkerThreadID();
                printf("[%d] BgMessageReplay> Emulation replay finished at %f seconds due to end of log.\n", mype, CmiWallTimer());
                printf("[%d] BgMessageReplay> Replayed %d local records and %d remote records, total of %ld bytes of data replayed.\n", mype, lcount, rcount, ftell(f));
       }
public:
        BgMessageReplay(FILE * f_, int node);
        ~BgMessageReplay() {
		done();
 		fclose(f);
	}
        bool record(char *msg) { return false; }
        int replay(void) {
                int nextPE;
                int ret =  fread(&nextPE, sizeof(int), 1, f);
                if (-1 == ret || ret == 0) {
			done();
                        ConverseExit();
                        return 0;
                }
		int mype = BgGetGlobalWorkerThreadID();
                if ( (nodelevel == 0 && nextPE == mype) ||
                     (nodelevel == 1 && nextPE/BgGetNumWorkThread() == mype/BgGetNumWorkThread()) ) {
//printf("BgMessageReplay> local message\n");
                  lcount ++;
                  return 0;
                }
                int nextSize;
                ret = fread(&nextSize, sizeof(int), 1, f);
                CmiAssert(ret ==1);
                CmiAssert(nextSize > 0);
                char *msg = (char*)CmiAlloc(nextSize);
                ret = fread(msg, sizeof(char), nextSize, f);
                if (ret != nextSize) {
                  CmiPrintf("Bigsim replay> fread returns only %d when asked %d bytes!\n", ret, nextSize);
                CmiAssert(ret == nextSize);
                }
                CmiAssert(CmiBgMsgLength(msg) == nextSize);
                // CmiPrintf("BgMessageReplay>  pe:%d size: %d handle: %d msg: %p\n", nextPE, nextSize, CmiBgMsgHandle(msg), msg);
                BgSendLocalPacket(mype%BgGetNumWorkThread(), CmiBgMsgHandle(msg), LARGE_WORK, nextSize, msg);
                rcount ++;
                return 1;
        }
};


extern void BgRead_nodeinfo(int node, int &startpe, int &endpe);

#endif
