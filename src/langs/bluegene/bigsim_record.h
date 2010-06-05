#ifndef BIGSIM_RECORD_H
#define BIGSIM_RECORD_H

extern void callAllUserTracingFunction();

extern int _heter;

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
        virtual CmiBool record(char *msg) { return CmiFalse; }
        virtual int replay() { return 0; }
	virtual void rewind() {}
};

class BgMessageRecorder : public BgMessageWatcher {
        FILE *f;
	long pos;
public:
        BgMessageRecorder(FILE * f_, int node);
        ~BgMessageRecorder() { fclose(f); }

        virtual CmiBool record(char *msg);
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
                printf("[%d] BgMessageReplay> Replayed %d local records and %d remote records, total of %lld bytes of data replayed.\n", mype, lcount, rcount, ftell(f));
       }
public:
        BgMessageReplay(FILE * f_, int node);
        ~BgMessageReplay() {
		done();
 		fclose(f);
	}
        CmiBool record(char *msg) { return CmiFalse; }
        int replay(void);
};


extern void BgRead_nodeinfo(int node, int &startpe, int &endpe);

#endif
