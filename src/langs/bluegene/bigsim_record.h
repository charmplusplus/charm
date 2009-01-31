#ifndef BIGSIM_RECORD_H
#define BIGSIM_RECORD_H

/// Message watcher: for record/replay support
class BgMessageWatcher {
public:
        virtual ~BgMessageWatcher() {}
        /**
         * This message is about to be processed by Charm.
         * If this function returns false, the message will not be processed.
         */
        virtual CmiBool record(char *msg) { return CmiFalse; }
        virtual int replay() { return 0; }
};

class BgMessageRecorder : public BgMessageWatcher {
        FILE *f;
public:
        BgMessageRecorder(FILE * f_);
        ~BgMessageRecorder() { fclose(f); }

        virtual CmiBool record(char *msg) {
                //if (BgGetGlobalWorkerThreadID()==1) printf("srcpe: %d size: %d handle: %d\n",CmiBgMsgSrcPe(msg),CmiBgMsgLength(msg),CmiBgMsgHandle(msg));
                int d = CmiBgMsgSrcPe(msg);
                fwrite(&d, sizeof(int), 1, f);
                if (d == BgGetGlobalWorkerThreadID()) return CmiTrue; // don't record local msg
                d = CmiBgMsgLength(msg);
                fwrite(&d, sizeof(int), 1, f);
/*
if (BgGetGlobalWorkerThreadID()==1 && CmiBgMsgHandle(msg) == 21) {
int *m = (int *) ((char *)msg+CmiReservedHeaderSize);
printf("replay: %d %d\n", m[0], m[1]);
}
*/
                fwrite(msg, sizeof(char), d, f);
                return CmiTrue;
        }
        virtual int replay() { return 0; }
};

class BgMessageReplay : public BgMessageWatcher {
        FILE * f;
        /// Read the next message we need from the file:
public:
        BgMessageReplay(FILE * f_);
        ~BgMessageReplay() {fclose(f);}
        CmiBool record(char *msg) { return CmiFalse; }
        int replay(void) {
                int nextPE;
                int ret =  fread(&nextPE, sizeof(int), 1, f);
                if (-1 == ret || ret == 0) {
                        ConverseExit();
                        return 0;
                }
                if (nextPE == BgGetGlobalWorkerThreadID()) {
//printf("BgMessageReplay> local message\n");
                  return 0;
                }
                int nextSize;
                ret = fread(&nextSize, sizeof(int), 1, f);
                CmiAssert(ret ==1);
                CmiAssert(nextSize > 0);
                char *msg = (char*)CmiAlloc(nextSize);
                ret = fread(msg, sizeof(char), nextSize, f);
                CmiAssert(ret == nextSize);
                CmiAssert(CmiBgMsgLength(msg) == nextSize);
//                CmiPrintf("BgMessageReplay>  pe:%d size: %d handle: %d msg: %p\n", nextPE, nextSize, CmiBgMsgHandle(msg), msg);
                BgSendLocalPacket(ANYTHREAD, CmiBgMsgHandle(msg), LARGE_WORK, nextSize, msg);
                return 1;
        }
};



#endif
