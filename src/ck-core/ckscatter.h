#ifndef __CKSCATTER_H
#define __CKSCATTER_H

#include "pup.h"
#include "envelope.h"

#define SCATTER(...) CkScatterWrapper(__VA_ARGS__)

class CkScatterWrapper{
	public:
		void *buf, *dest;
		int *disp, *cnt;
		int ndest, size;

		void pup(PUP::er &p){
			pup_bytes(&p, this, sizeof(CkScatterWrapper));
		}

		void setsize(int _size) {
			size = _size;
		}

		void setoffset(size_t _offset) {
			buf = (void *)(_offset);
		}

		void* getbuf(void *impl_buf){
			//CkPrintf("getbuf, size: %d \n", (size_t)buf);
			return ((char *)impl_buf) + ((size_t)buf);
		}  

		CkScatterWrapper() : buf(NULL), disp(NULL), dest(NULL), cnt(NULL), ndest(0) {}
		CkScatterWrapper(void *_buf, int _ndest, int *_disp, void *_dest, int *_cnt) : buf(_buf), disp(_disp), dest(_dest), cnt(_cnt), ndest(_ndest){}
		CkScatterWrapper(void *_buf, int _ndest, int *_disp, int *_cnt) : buf(_buf), disp(_disp), cnt(_cnt), ndest(_ndest), dest(NULL){}
};

void getScatterInfo(void *msg, CkScatterWrapper *w); 

void* createScatterMsg(void *msg, CkScatterWrapper &w, int ind);

#endif
