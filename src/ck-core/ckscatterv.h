#ifndef __CKSCATTERV_H
#define __CKSCATTERV_H

#include "pup.h"
#include "envelope.h"

#define SCATTERV(...) CkScattervWrapper(__VA_ARGS__)

class CkScattervWrapper{
	public:
		void *buf, *dest;
		int *disp, *cnt;
		int ndest, size;

		void pup(PUP::er &p){
			pup_bytes(&p, this, sizeof(CkScattervWrapper));
		}

		inline void setSize(int _size) {
			size = _size;
		}

		inline void setOffset(size_t _offset) {
			buf = (void *)(_offset);
		}

		inline void* getBuf(void *impl_buf) const{
			//CkPrintf("getBuf, size: %d \n", (size_t)buf);
			return ((char *)impl_buf) + ((size_t)buf);
		}

		CkScattervWrapper() : buf(NULL), disp(NULL), dest(NULL), cnt(NULL), ndest(0) {}
		CkScattervWrapper(void *_buf, int _ndest, int *_disp, void *_dest, int *_cnt) : buf(_buf), disp(_disp), dest(_dest), cnt(_cnt), ndest(_ndest){}
		CkScattervWrapper(void *_buf, int _ndest, int *_disp, int *_cnt) : buf(_buf), disp(_disp), cnt(_cnt), ndest(_ndest), dest(NULL){}
};

void getScattervInfo(void *msg, CkScattervWrapper *w);

void* createScattervMsg(void *msg, CkScattervWrapper &w, int ind);

#endif
