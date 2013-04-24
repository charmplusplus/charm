#ifndef PACKLIB_H

#define PACKLIB_H

#include <stdio.h>
#include <stdlib.h>
#include <converse.h>

enum PackErr { EOk = 0, EOverflow = -1, EType = -2 };

class Packer {
public:
  enum ItemType { t_voidp, t_char, t_uchar, t_int, t_uint, 
		  t_long, t_ulong, t_float, t_double };

  Packer(bool _debug = false) { 
    head = tail = 0; bytes = 0;  debug = _debug;
  };

  ~Packer() {
    item* i = head;
    item* nxt;
    while (i != 0) { nxt = i->nxt; delete i; i = nxt; }
  }

  void pack(const char i) { enqueue(new item(i)); };
  void pack(const unsigned char i) { enqueue(new item(i)); };
  void pack(const int i) { enqueue(new item(i)); };
  void pack(const unsigned int i) { enqueue(new item(i)); };
  void pack(const long i) { enqueue(new item(i)); };
  void pack(const unsigned long i) { enqueue(new item(i)); };
  void pack(const float i) { enqueue(new item(i)); };
  void pack(const double i) { enqueue(new item(i)); };

  void pack(const char *const i, const int count) {
    enqueue(new item(i,count));
  };
  void pack(const unsigned char *const i, const int count) { 
    enqueue(new item(i,count)); 
  };
  void pack(const int *const i, const int count) {
    enqueue(new item(i,count));
  };
  void pack(const unsigned int *const i, int count) {
    enqueue(new item(i,count));
  };
  void pack(const long *const i, const int count) {
    enqueue(new item(i,count));
  };
  void pack(const unsigned long *const i, const int count) { 
    enqueue(new item(i,count));
  };
  void pack(const float *const i, const int count) { 
    enqueue(new item(i,count));
  };
  void pack(const double *const i, const int count) {
    enqueue(new item(i,count));
  };

  int buffer_size() const { return bytes + sizeof(int); };

  PackErr fill_buffer(void *const buffer, int buf_bytes) {
#if CMK_CPP_CAST_LEGAL
    char* buf_ptr = static_cast<char*>(buffer);
    if (buf_bytes >= sizeof(int))
      *(static_cast<int*>(buffer)) = buf_bytes;
#else
    char* buf_ptr = (char*)buffer;
    if (buf_bytes >= sizeof(int))
      *((int*)buffer) = buf_bytes;
#endif

    buf_ptr += sizeof(int);
    buf_bytes -= sizeof(int);
      
    item* cur_item;
    while (cur_item = dequeue()) {
      const void* item_ptr;

#if CMK_CPP_CAST_LEGAL
      if (cur_item->type == t_char) {
	item_ptr = static_cast<const void*>(&cur_item->cdat);
      } else if (cur_item->type == t_uchar) {
	item_ptr = static_cast<const void*>(&cur_item->ucdat);
      } else if (cur_item->type == t_int) {
	item_ptr = static_cast<const void*>(&cur_item->idat);
      } else if (cur_item->type == t_uint) {
	item_ptr = static_cast<const void*>(&cur_item->uidat);
      } else if (cur_item->type == t_long) {
	item_ptr = static_cast<const void*>(&cur_item->ldat);
      } else if (cur_item->type == t_ulong) {
	item_ptr = static_cast<const void*>(&cur_item->uldat);
      } else if (cur_item->type == t_float) {
	item_ptr = static_cast<const void*>(&cur_item->fdat);
      } else if (cur_item->type == t_double) {
	item_ptr = static_cast<const void*>(&cur_item->ddat);
      } else if (cur_item->type == t_voidp) {
	item_ptr = static_cast<const void*>(cur_item->arr);
      } else return EType;
#else
      if (cur_item->type == t_char) {
	item_ptr = (const void*)(&cur_item->cdat);
      } else if (cur_item->type == t_uchar) {
	item_ptr = (const void*)(&cur_item->ucdat);
      } else if (cur_item->type == t_int) {
	item_ptr = (const void*)(&cur_item->idat);
      } else if (cur_item->type == t_uint) {
	item_ptr = (const void*)(&cur_item->uidat);
      } else if (cur_item->type == t_long) {
	item_ptr = (const void*)(&cur_item->ldat);
      } else if (cur_item->type == t_ulong) {
	item_ptr = (const void*)(&cur_item->uldat);
      } else if (cur_item->type == t_float) {
	item_ptr = (const void*)(&cur_item->fdat);
      } else if (cur_item->type == t_double) {
	item_ptr = (const void*)(&cur_item->ddat);
      } else if (cur_item->type == t_voidp) {
	item_ptr = (const void*)(cur_item->arr);
      } else return EType;
#endif

      const int ibytes = cur_item->size;
      if (debug && buf_bytes >= sizeof(int)) {
#if CMK_CPP_CAST_LEGAL
	const char* cptr = reinterpret_cast<const char*>(&ibytes);
#else
	const char* cptr = (const char*)(&ibytes);
#endif
	for(int i=0; i < sizeof(int); i++)
	  *buf_ptr++ = *cptr++;
	buf_bytes -= sizeof(int);
      }
      
#if CMK_CPP_CAST_LEGAL
      const char* cptr = static_cast<const char*>(item_ptr);
#else
      const char* cptr = (const char*)(item_ptr);
#endif
      if (buf_bytes >= ibytes) {
	for(int i=0; i < ibytes; i++) {
	  *buf_ptr++ = *cptr++;
	}
	buf_bytes -= ibytes;
      } else return EOverflow;

      delete cur_item;
    }
    return EOk;
  }

private:
  class item {
  public:
    item(const char i) { 
      size = sizeof(char);
      type = t_char; 
      cdat = i; 
      nxt = 0; 
    };

    item(const unsigned char i) { 
      size = sizeof(unsigned char);
      type = t_uchar; 
      ucdat = i; 
      nxt = 0; 
    };

    item(const int i) { 
      size = sizeof(int);
      type = t_int; 
      idat = i; 
      nxt = 0; 
    };

    item(const unsigned int i) { 
      size = sizeof(unsigned int);
      type = t_uint; 
      uidat = i; 
      nxt = 0; 
    };

    item(const long i) { 
      size = sizeof(long int);
      type = t_long; 
      ldat = i; 
      nxt = 0; 
    };

    item(const unsigned long i) { 
      size = sizeof(unsigned long int);
      type = t_ulong; 
      uldat = i; 
      nxt = 0; 
    };

    item(const float i) { 
      size = sizeof(float);
      type = t_float; 
      fdat = i; 
      nxt = 0; 
    };

    item(const double i) { 
      size = sizeof(double);
      type = t_double; 
      ddat = i; 
      nxt = 0; 
    };

    item(const char *i, const int _size) { 
      size = _size * sizeof(char);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const unsigned char *i, const int _size) { 
      size = _size * sizeof(unsigned char);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const int *i, const int _size) { 
      size = _size * sizeof(int);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const unsigned int *i, const int _size) { 
      size = _size * sizeof(unsigned int);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const long *i, const int _size) { 
      size = _size * sizeof(long);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const unsigned long *i, const int _size) { 
      size = _size * sizeof(unsigned long);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const float *i, const int _size) { 
      size = _size * sizeof(float);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    item(const double *i, const int _size) { 
      size = _size * sizeof(double);
      type = t_voidp;
#if CMK_CPP_CAST_LEGAL
      arr = static_cast<const void*>(i); 
#else
      arr = (const void*)(i); 
#endif
      nxt = 0; 
    };

    int size;
    ItemType type;
    union {
      const void* arr;
      char cdat;
      unsigned char ucdat;
      int idat;
      unsigned int uidat;
      long ldat;
      unsigned long uldat;
      float fdat;
      double ddat;
    };
    item* nxt;
  };

  void enqueue(item* _item) {
    if (head==0)
      head = tail = _item;
    else {
      tail->nxt = _item;
      tail = tail->nxt;
    }
    bytes += _item->size;
    if (debug) bytes += sizeof(int);
  };

  item* dequeue() {
    item* ret = head;
    if (ret != 0) {
      head = ret->nxt;
      bytes -= ret->size;
      if (debug) bytes -= sizeof(int);
    }
    return ret;
  };
  
  item* head;
  item* tail;
  int bytes;
  bool debug;
};

class Unpacker {
public:
  Unpacker(const void *const _buffer, bool _debug = false) {
    buffer = _buffer;
#if CMK_CPP_CAST_LEGAL
    bufsz = *(static_cast<const int*>(buffer)) - sizeof(int);
    buf_ptr = static_cast<const char*>(buffer) + sizeof(int);
#else
    bufsz = *((const int*)(buffer)) - sizeof(int);
    buf_ptr = (const char*)(buffer) + sizeof(int);
#endif
    unpacked = sizeof(int);
    debug = _debug;
  };

#if CMK_CPP_CAST_LEGAL
  PackErr unpack(char *i) { 
    return unpack_item(static_cast<void*>(i), sizeof(char));
  };
  PackErr unpack(unsigned char *i) { 
    return unpack_item(static_cast<void*>(i), sizeof(unsigned char));
  };
  PackErr unpack(int *i) { 
    return unpack_item(static_cast<void*>(i), sizeof(int));
  };
  PackErr unpack(unsigned int *i) {
    return unpack_item(static_cast<void*>(i), sizeof(unsigned int));
  };
  PackErr unpack(long *i) { 
    return unpack_item(static_cast<void*>(i), sizeof(long));
  };
  PackErr unpack(unsigned long *i) {
    return unpack_item(static_cast<void*>(i), sizeof(unsigned long));
  };
  PackErr unpack(float *i) {
    return unpack_item(static_cast<void*>(i), sizeof(float));
  };
  PackErr unpack(double *i) {
    return unpack_item(static_cast<void*>(i), sizeof(double));
  };
  PackErr unpack(char *i,const int items) { 
    return unpack_item(static_cast<void*>(i), sizeof(char)*items);
  };
  PackErr unpack(unsigned char *i,const int items) { 
    return unpack_item(static_cast<void*>(i), 
		       sizeof(unsigned char)*items);
  };
  PackErr unpack(int *i,const int items) { 
    return unpack_item(static_cast<void*>(i), sizeof(int)*items);
  };
  PackErr unpack(unsigned int *i,const int items) {
    return unpack_item(static_cast<void*>(i), 
		       sizeof(unsigned int)*items);
  };
  PackErr unpack(long *i,const int items) { 
    return unpack_item(static_cast<void*>(i), sizeof(long)*items);
  };
  PackErr unpack(unsigned long *i,const int items) {
    return unpack_item(static_cast<void*>(i),
		       sizeof(unsigned long)*items);
  };
  PackErr unpack(float *i,const int items) {
    return unpack_item(static_cast<void*>(i), sizeof(float)*items);
  };
  PackErr unpack(double *i,const int items) {
    return unpack_item(static_cast<void*>(i), sizeof(double)*items);
  };
#else
  PackErr unpack(char *i) { 
    return unpack_item((void*)(i), sizeof(char));
  };
  PackErr unpack(unsigned char *i) { 
    return unpack_item((void*)(i), sizeof(unsigned char));
  };
  PackErr unpack(int *i) { 
    return unpack_item((void*)(i), sizeof(int));
  };
  PackErr unpack(unsigned int *i) {
    return unpack_item((void*)(i), sizeof(unsigned int));
  };
  PackErr unpack(long *i) { 
    return unpack_item((void*)(i), sizeof(long));
  };
  PackErr unpack(unsigned long *i) {
    return unpack_item((void*)(i), sizeof(unsigned long));
  };
  PackErr unpack(float *i) {
    return unpack_item((void*)(i), sizeof(float));
  };
  PackErr unpack(double *i) {
    return unpack_item((void*)(i), sizeof(double));
  };
  PackErr unpack(char *i,const int items) { 
    return unpack_item((void*)(i), sizeof(char)*items);
  };
  PackErr unpack(unsigned char *i,const int items) { 
    return unpack_item((void*)(i), 
		       sizeof(unsigned char)*items);
  };
  PackErr unpack(int *i,const int items) { 
    return unpack_item((void*)(i), sizeof(int)*items);
  };
  PackErr unpack(unsigned int *i,const int items) {
    return unpack_item((void*)(i), 
		       sizeof(unsigned int)*items);
  };
  PackErr unpack(long *i,const int items) { 
    return unpack_item((void*)(i), sizeof(long)*items);
  };
  PackErr unpack(unsigned long *i,const int items) {
    return unpack_item((void*)(i),
		       sizeof(unsigned long)*items);
  };
  PackErr unpack(float *i,const int items) {
    return unpack_item((void*)(i), sizeof(float)*items);
  };
  PackErr unpack(double *i,const int items) {
    return unpack_item((void*)(i), sizeof(double)*items);
  };
#endif
  int bytes_unpacked() { return unpacked; };

 private:
  PackErr unpack_item(void *item, int bytes) {
    if (debug) {
      int item_sz;
#if CMK_CPP_CAST_LEGAL
      char* item_ptr = reinterpret_cast<char*>(&item_sz);
#else
      char* item_ptr = (char*)(&item_sz);
#endif
      if (bufsz >= sizeof(int)) {
	for(int i=0; i<sizeof(int); i++)
	  *item_ptr++ = *buf_ptr++;
	bufsz -= sizeof(int);
	unpacked += sizeof(int);
      }
      if (item_sz != bytes) {
	printf("Unpack mismatch, hanging\n");
	while (1) ;
      }
    }

#if CMK_CPP_CAST_LEGAL    
    char* item_ptr = static_cast<char*>(item);
#else
    char* item_ptr = (char*)(item);
#endif
    if (bufsz >= bytes) {
      for(int i=0; i<bytes; i++)
	*item_ptr++ = *buf_ptr++;
      bufsz -= bytes;
      unpacked += bytes;
    } else return EOverflow;
    return EOk;
  };

  const void* buffer;
  const char* buf_ptr;
  int bufsz;
  int unpacked;
  bool debug;
};

#endif /* PACKLIB_H */
