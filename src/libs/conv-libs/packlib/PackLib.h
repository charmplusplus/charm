#ifndef PACKLIB_H

#define PACKLIB_H

enum PackErr { EOk = 0, EOverflow = -1, EType = -2 };

class Packer {
public:
  enum ItemType { t_voidp, t_char, t_uchar, t_int, t_uint, 
		  t_long, t_ulong, t_float, t_double };

  Packer() { head = tail = 0; bytes = 0; };

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
    char* buf_ptr = static_cast<char *const>(buffer);
    if (buf_bytes >= sizeof(int))
      *(static_cast<int*>(buffer)) = buf_bytes;
    buf_ptr += sizeof(int);
    buf_bytes -= sizeof(int);
      
    item* cur_item;
    int inum=0;
    while (cur_item = dequeue()) {
      const void* item_ptr;

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
      const int ibytes = cur_item->size;
      const char* cptr = static_cast<const char*>(item_ptr);
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

    item(const char *const i, const int _size) { 
      size = _size * sizeof(char);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const unsigned char *const i, const int _size) { 
      size = _size * sizeof(unsigned char);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const int *const i, const int _size) { 
      size = _size * sizeof(int);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const unsigned int *const i, const int _size) { 
      size = _size * sizeof(unsigned int);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const long *const i, const int _size) { 
      size = _size * sizeof(long);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const unsigned long *const i, const int _size) { 
      size = _size * sizeof(unsigned long);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const float *const i, const int _size) { 
      size = _size * sizeof(float);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
      nxt = 0; 
    };

    item(const double *const i, const int _size) { 
      size = _size * sizeof(double);
      type = t_voidp;
      arr = static_cast<const void*>(i); 
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
  };

  item* dequeue() {
    item* ret = head;
    if (ret != 0) {
      head = ret->nxt;
      bytes -= ret->size;
    }
    return ret;
  };
  
  item* head;
  item* tail;
  int bytes;
};

class Unpacker {
public:
  Unpacker(const void *const _buffer) {
    buffer = _buffer;
    bufsz = *(static_cast<const int *const>(buffer)) - sizeof(int);
    buf_ptr = static_cast<const char*>(buffer) + sizeof(int);
  };
    
  PackErr unpack(char *const i) { 
    return unpack_item(static_cast<void *const>(i), sizeof(char));
  };
  PackErr unpack(unsigned char *const i) { 
    return unpack_item(static_cast<void *const>(i), sizeof(unsigned char));
  };
  PackErr unpack(int *const i) { 
    return unpack_item(static_cast<void *const>(i), sizeof(int));
  };
  PackErr unpack(unsigned int *const i) {
    return unpack_item(static_cast<void *const>(i), sizeof(unsigned int));
  };
  PackErr unpack(long *const i) { 
    return unpack_item(static_cast<void *const>(i), sizeof(long));
  };
  PackErr unpack(unsigned long *const i) {
    return unpack_item(static_cast<void *const>(i), sizeof(unsigned long));
  };
  PackErr unpack(float *const i) {
    return unpack_item(static_cast<void *const>(i), sizeof(float));
  };
  PackErr unpack(double *const i) {
    return unpack_item(static_cast<void *const>(i), sizeof(double));
  };
  PackErr unpack(char *const i,const int items) { 
    return unpack_item(static_cast<void *const>(i), sizeof(char)*items);
  };
  PackErr unpack(unsigned char *const i,const int items) { 
    return unpack_item(static_cast<void *const>(i), 
		       sizeof(unsigned char)*items);
  };
  PackErr unpack(int *const i,const int items) { 
    return unpack_item(static_cast<void *const>(i), sizeof(int)*items);
  };
  PackErr unpack(unsigned int *const i,const int items) {
    return unpack_item(static_cast<void *const>(i), 
		       sizeof(unsigned int)*items);
  };
  PackErr unpack(long *const i,const int items) { 
    return unpack_item(static_cast<void *const>(i), sizeof(long)*items);
  };
  PackErr unpack(unsigned long *const i,const int items) {
    return unpack_item(static_cast<void *const>(i),
		       sizeof(unsigned long)*items);
  };
  PackErr unpack(float *const i,const int items) {
    return unpack_item(static_cast<void *const>(i), sizeof(float)*items);
  };
  PackErr unpack(double *const i,const int items) {
    return unpack_item(static_cast<void *const>(i), sizeof(double)*items);
  };

 private:
  PackErr unpack_item(void *const item, int bytes) {
    char* item_ptr = static_cast<char*>(item);
    
    if (bufsz >= bytes) {
      for(int i=0; i<bytes; i++)
	*item_ptr++ = *buf_ptr++;
      bufsz -= bytes;
    } else return EOverflow;
    return EOk;
  };

  const void* buffer;
  const char* buf_ptr;
  int bufsz;
};

#endif /* PACKLIB_H */
