#ifndef MSTREAM_H
#define MSTREAM_H

struct StreamMessage {
  char header[CmiMsgHeaderSizeBytes];
  int PE;
  int tag;
  unsigned short len; // sizeof the data 
  unsigned short isLast; // 1 if its last packet
  char data[1];
};

class Communicate;

class MIStream {
  private:
    int PE, tag;
    StreamMessage *msg;
    int currentPos;
    Communicate *cobj;
    MIStream *Get(char *buf, int len);  // get len bytes from message to buf
  public:
    MIStream(Communicate *c, int pe, int tag);
    ~MIStream();
    MIStream *get(char &data) { 
      return Get(&data,sizeof(char)); 
    }
    MIStream *get(unsigned char &data) { 
      return Get((char *)&data,sizeof(unsigned char)); 
    }
    MIStream *get(short &data) { 
      return Get((char *)&data, sizeof(short)); 
    }
    MIStream *get(unsigned short &data) { 
      return Get((char *)&data, sizeof(unsigned short)); 
    }
    MIStream *get(int &data) { 
      return Get((char *)&data, sizeof(int)); 
    }
    MIStream *get(unsigned int &data) { 
      return Get((char *)&data, sizeof(unsigned int)); 
    }
    MIStream *get(long &data) { 
      return Get((char *)&data, sizeof(long)); 
    }
    MIStream *get(unsigned long &data) { 
      return Get((char *)&data, sizeof(unsigned long)); 
    }
    MIStream *get(float &data) { 
      return Get((char *)&data, sizeof(float)); 
    }
    MIStream *get(double &data) { 
      return Get((char *)&data, sizeof(double)); 
    }
    MIStream *get(int len, char *data) { 
      return Get(data,len*sizeof(char)); 
    }
    MIStream *get(int len, unsigned char *data) { 
      return Get((char *)data,len*sizeof(unsigned char)); 
    }
    MIStream *get(int len, short *data) { 
      return Get((char *)data,len*sizeof(short)); 
    }
    MIStream *get(int len, unsigned short *data) { 
      return Get((char *)data,len*sizeof(unsigned short)); 
    }
    MIStream *get(int len, int *data) { 
      return Get((char *)data,len*sizeof(int)); 
    }
    MIStream *get(int len, unsigned int *data) { 
      return Get((char *)data,len*sizeof(unsigned int)); 
    }
    MIStream *get(int len, long *data) { 
      return Get((char *)data,len*sizeof(long)); 
    }
    MIStream *get(int len, unsigned long *data) { 
      return Get((char *)data,len*sizeof(unsigned long)); 
    }
    MIStream *get(int len, float *data) { 
      return Get((char *)data,len*sizeof(float)); 
    }
    MIStream *get(int len, double *data) { 
      return Get((char *)data,len*sizeof(double)); 
    }
};

class MOStream {
  private:
    int PE, tag;
    unsigned int bufLen;
    StreamMessage *msgBuf;
    Communicate *cobj;
    MOStream *Put(char *buf, int len);  // put len bytes from buf into message
  public:
    MOStream(Communicate *c, int pe, int tag, unsigned int bufSize);
    ~MOStream();
    void end(void);
    MOStream *put(char data) { 
      return Put(&data,sizeof(char)); 
    }
    MOStream *put(unsigned char data) { 
      return Put((char *)&data,sizeof(unsigned char)); 
    }
    MOStream *put(short data) { 
      return Put((char *)&data, sizeof(short)); 
    }
    MOStream *put(unsigned short data) { 
      return Put((char *)&data, sizeof(unsigned short)); 
    }
    MOStream *put(int data) { 
      return Put((char *)&data, sizeof(int)); 
    }
    MOStream *put(unsigned int data) { 
      return Put((char *)&data, sizeof(unsigned int)); 
    }
    MOStream *put(long data) { 
      return Put((char *)&data, sizeof(long)); 
    }
    MOStream *put(unsigned long data) { 
      return Put((char *)&data, sizeof(unsigned long)); 
    }
    MOStream *put(float data) { 
      return Put((char *)&data, sizeof(float)); 
    }
    MOStream *put(double data) { 
      return Put((char *)&data, sizeof(double)); 
    }
    MOStream *put(int len, char *data) { 
      return Put(data,len*sizeof(char)); 
    }
    MOStream *put(int len, unsigned char *data) { 
      return Put((char *)data,len*sizeof(unsigned char)); 
    }
    MOStream *put(int len, short *data) { 
      return Put((char *)data,len*sizeof(short)); 
    }
    MOStream *put(int len, unsigned short *data) { 
      return Put((char *)data,len*sizeof(unsigned short)); 
    }
    MOStream *put(int len, int *data) { 
      return Put((char *)data,len*sizeof(int)); 
    }
    MOStream *put(int len, unsigned int *data) { 
      return Put((char *)data,len*sizeof(unsigned int)); 
    }
    MOStream *put(int len, long *data) { 
      return Put((char *)data,len*sizeof(long)); 
    }
    MOStream *put(int len, unsigned long *data) { 
      return Put((char *)data,len*sizeof(unsigned long)); 
    }
    MOStream *put(int len, float *data) { 
      return Put((char *)data,len*sizeof(float)); 
    }
    MOStream *put(int len, double *data) { 
      return Put((char *)data,len*sizeof(double)); 
    }
};

#endif
