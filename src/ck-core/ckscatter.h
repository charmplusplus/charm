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
		CkArrayID aid;

		void pup(PUP::er &p){
			pup_bytes(&p, this, sizeof(CkScatterWrapper));
		}

		void setsize(int _size) {
			size = _size;
		}

		void setoffset_buf(size_t _offset) {
			buf = (void *)(_offset);
		}

		void setoffset_disp(size_t _offset) {
			disp = (int *)(_offset);
		}

		void setoffset_dest(size_t _offset) {
			dest = (void *)(_offset);
		}

		void setoffset_cnt(size_t _offset) {
			cnt = (int *)(_offset);
		}

		void* getbuf(void *impl_buf){
			//CkPrintf("getbuf, size: %d \n", (size_t)buf);
			return ((char *)impl_buf) + ((size_t)buf);
		}  

		void* getdisp(void *impl_buf){
			//CkPrintf("getdisp, size: %d \n", (size_t)disp);
			return ((char *)impl_buf) + ((size_t)disp);
		}

		void* getdest(void *impl_buf){
			//CkPrintf("getdest, size: %d \n", (size_t)dest);
			return ((char *)impl_buf) + ((size_t)dest);
		}

		void* getcnt(void *impl_buf){
			//CkPrintf("getcnt, size: %d \n", (size_t)cnt);
			return ((char *)impl_buf) + ((size_t)cnt);
		}

		void unpackInfo(void *impl_buf){
		   buf = getbuf(impl_buf);
		   disp = (int *)getdisp(impl_buf);
		   dest = (void *)getdest(impl_buf);
		   cnt = (int *)getcnt(impl_buf);
		}

        void packInfo(void *impl_buf){
            //CkPrintf("CkScatterWrapper, packInfo, buf: %p, disp: %p, dest: %p, cnt: %p \n", buf, disp, dest, cnt);
            buf = (void *) ((size_t)(((char *)buf) - ((char *)impl_buf)));
            disp = (int *) ((size_t)(((char *)disp) - ((char *)impl_buf)));
            dest = (void *)((size_t)(((char *)dest) - ((char *)impl_buf)));
            cnt = (int *)  ((size_t)(((char *)cnt) - ((char *)impl_buf)));
        }

		CkScatterWrapper() : buf(NULL), disp(NULL), dest(NULL), cnt(NULL), ndest(0) {}
		CkScatterWrapper(void *_buf, int _ndest, int *_disp, void *_dest, int *_cnt) : buf(_buf), disp(_disp), dest(_dest), cnt(_cnt), ndest(_ndest){}
		CkScatterWrapper(void *_buf, int _ndest, int *_disp, int *_cnt) : buf(_buf), disp(_disp), cnt(_cnt), ndest(_ndest), dest(NULL){}
};

void getScatterInfo(void *msg, CkScatterWrapper *w); 

void* createScatterMsg(void *msg, CkScatterWrapper &w, int ind);

void* createScatterMsg(void *msg, CkScatterWrapper &w, std::vector<int> &indices);

#endif
