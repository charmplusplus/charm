#ifndef CHARISMA_H
#define CHARISMA_H

#include "charisma.decl.h"

class Name
{
  private:
    char *curname;
    int curbuflen;
    int curlen;
  public:
    Name(void)
    {
      curbuflen = 100;
      curname = new char[curbuflen];
      curname[0] = '\0';
      curlen = 0;
    }
    void add(const char *n)
    {
      int len = strlen(n);
      if((curlen+len+2) > curbuflen) {
        char *t = new char[curlen+len+2];
        strcpy(t, curname);
        delete[] curname;
        curname = t;
        curbuflen = curlen+len+2;
      }
      if(curlen!=0) {
        strcat(curname, ".");
        curlen++;
      }
      strcat(curname, n);
      curlen += len;
    }
    void remove(const char *n)
    {
      int len = strlen(n);
      curlen = curlen-len;
      curname[curlen] = '\0';
      if(curlen!=0) {
        curname[--curlen] = '\0';
      }
    }
    inline operator char *() const { return curname; }
    ~Name()
    {
      delete[] curname;
    }
};

class Component
{
  private:
    char *name;
    char *type;
    int idx; // unique index for every component given for partitioning
    int pe; // mapping resulting from partitioning
    int nout; // number of output ports
    int nin; // number of input ports
    int ncout; // number of connected output ports
    int ncin; // number of connected input ports
  public:
    Component(const char *n, const char *t)
    {
      name = new char[strlen(n)+1];
      strcpy(name,n);
      type = new char[strlen(t)+1];
      strcpy(type,t);
      ncout = ncin = nout = nin = 0;
      idx = pe = (-1); // something invalid
    }
    ~Component()
    {
      delete[] type;
    }
    char *getName(void) { return name; }
    char *getType(void) { return type; }
    void setIdx(int _idx) { idx = _idx; }
    int getIdx(void) { return idx; }
    void setPe(int _pe) { pe = _pe; }
    int getPe(void) { return pe; }
    void addOutPort(void) { nout++; }
    void addInPort(void) { nin++; }
    void connectOutPort(void) { ncout++; }
    void connectInPort(void) { ncin++; }
    int getNumPorts(void) { return nout + nin; }
    int getNumConnectedPorts(void) { return ncout + ncin; }
};

#define CHARISMA_IN 1
#define CHARISMA_OUT 2

class Port
{
  private:
    int inout; // IN/OUT
    char *name;
    Component *comp;
    char *type;
    Port *peer;
  public:
    Port(const char *n, Component *c, const char *t, int io)
    {
      name = new char[strlen(n)+1];
      strcpy(name,n);
      comp = c;
      type = new char[strlen(t)+1];
      strcpy(type,t);
      inout = io;
      peer = 0;
    }
    void setPeer(Port *p)
    {
      peer = p;
    }
    ~Port()
    {
      delete[] name;
      delete[] type;
    }
    char *getName(void) { return name; }
    char *getType(void) { return type; }
    Port *getPeer(void) { return peer; }
    Component *getComp(void) { return comp; }
};

extern "C"
void METIS_PartGraphRecursive(int* n, int* xadj, int* adjncy, int* vwght,
    int* adjwgt, int* wgtflag, int* numflag, int* nparts, int* options,
    int* edgecut, int* part);

class CharismaGraph
{
  private:
    CkHashtableTslow<const char *, Port *> ports;
    CkHashtableTslow<const char *, Component *> comps;
    Name *curname;
  public:
    CharismaGraph(void)
    {
      curname = new Name;
    }
    // indicate that all the following registrations belong to component
    // named name.
    void start(const char *name, const char *type)
    {
      curname->add(name);
      CkPrintf("starting registration for %s\n", (char *) curname);
      Component *curcomp = new Component((const char *)curname, type);
      comps.put((const char *)curname) = curcomp;
    }
    // end registrations for component called name
    void end(const char *name)
    {
      CkPrintf("ending registration for %s\n", (char *) curname);
      curname->remove(name);
    }
    // register an output port
    void registerOutPort(const char *name, const char *type)
    {
      Component *curcomp = comps.get((const char *) curname);
      curcomp->addOutPort();
      curname->add(name);
      CkPrintf("registering outport %s\n", (char *) curname);
      ports.put((const char *)curname) = new Port((const char *)curname,
          curcomp, type, CHARISMA_OUT);
      curname->remove(name);
    }
    // register an input port
    void registerInPort(const char *name, const char *type)
    {
      Component *curcomp = comps.get((const char *) curname);
      curcomp->addInPort();
      curname->add(name);
      CkPrintf("registering outport %s\n", (char *) curname);
      ports.put((const char *)curname) = new Port((const char *) curname,
          curcomp, type, CHARISMA_IN);
      curname->remove(name);
    }
    // connect an output port name1 to input port name2
    void connect(const char *name1, const char *name2)
    {
      curname->add(name1);
      char *oname = new char[strlen((char*)curname)+1];
      strcpy(oname, (char*) curname);
      curname->remove(name1);
      curname->add(name2);
      char *iname = new char[strlen((char*)curname)+1];
      strcpy(iname, (char*) curname);
      curname->remove(name2);
      Port *oport = ports.get(oname);
      Port *iport = ports.get(iname);
      if(strcmp(oport->getType(), iport->getType())!=0)
        CkAbort("Types of connected ports do not match !!!\n");
      oport->setPeer(iport);
      iport->setPeer(oport);
      oport->getComp()->connectOutPort();
      iport->getComp()->connectInPort();
      delete[] oname;
      delete[] iname;
    }
    void Partition(void)
    {
      int i;
      int n = 0; // number of objects
      CkHashtableIterator *ht = comps.iterator();
      CkHashtableIterator *pt = ports.iterator();
      // count the number of components, assigning unique index to each
      // component, this index will be used for partitioning
      ht->seekStart();
      while(ht->hasNext()) {
        Component *c = (Component *) ht->next();
        c->setIdx(n);
        n++;
      }
      int *nconnect = new int[n];
      int tadj = 0;
      ht->seekStart();
      while(ht->hasNext()) {
        Component *c = (Component *) ht->next();
        nconnect[c->getIdx()] = c->getNumConnectedPorts();
        tadj += nconnect[c->getIdx()];
      }
      // now the nconnect array contains the number of edges for each vertex
      // and tadj contains total number of connections
      int *xadj = new int[n];
      int *adjncy = new int[tadj];
      // construct the adjcency in CSR format as required by Metis
      xadj[0] = 0;
      for(i=1;i<n;i++)
        xadj[i] = xadj[i-1] + nconnect[i-1];
      for(i=0;i<tadj;i++)
        adjncy[i] = (-1); // set to something invalid
      pt->seekStart();
      while(pt->hasNext()) {
        Port *p = (Port *) pt->next();
        Port *peer = p->getPeer();
        if(peer==0) // an unconnected port
          continue;
        Component *comp1 = p->getComp();
        Component *comp2 = peer->getComp();
        int idx1 = comp1->getIdx();
        int idx2 = comp2->getIdx();
        int i;
        for(i=xadj[idx1];adjncy[i]!=(-1);i++);
        adjncy[i] = idx2;
      }
      int *vwgt = new int[n];
      int *adjwgt = new int[tadj];
      // currently all vertices are assumed to be of same wgt
      // this will change after embellishing the connection code
      for(i=0;i<n; i++)
        vwgt[i] = 1;
      // currently all vertices are assumed to be of same wgt
      // this will change after embellishing the connection code
      for(i=0;i<tadj; i++)
        adjwgt[i] = 1;
      int wgtflag = 3; // weights on both vertices and edges
      int numflag = 0; // we use C-style numbering (starting with 0)
      int nparts = CkNumPes(); // number of partitions desired
      int options[5];
      options[0] = 0; // use default options
      int edgecut; // contains edgecut after partitioning
      int *part = new int[n]; // contains mapping output to processor number
      METIS_PartGraphRecursive(&n, xadj, adjncy, vwgt, adjwgt, &wgtflag,
          &numflag, &nparts, options, &edgecut, part);
      ht->seekStart();
      while(ht->hasNext()) {
        Component* c = (Component *) ht->next();
        c->setPe(part[c->getIdx()]);
      }
    }
};

class CkArrayIndexCharisma: public CkArrayIndex
{
  int i, j, k;
  public:
    CkArrayIndexCharisma(int _i) {
      i = _i; j = 0; k = 0;
      nInts = 1;
    }
    CkArrayIndexCharisma(int _i, int _j) {
      i = _i; j= _j; k= 0;
      nInts = 2;
    }
    CkArrayIndexCharisma(int _i, int _j, int _k) {
      i = _i; j = _j; k= _k;
      nInts = 3;
    }
};

class CharismaInPort
{
  public:
    virtual void send(void *msg, int len) = 0;
    void _create(const char *name)
    {
      // todo: tell Charisma that the inport is created
    }
};

class CharismaOutPort
{
  protected:
    CharismaInPort *inport;
  public:
    virtual void emit(void *data, int len)
    {
      inport->send(data, len);
    }
    void _create(const char *name)
    {
      // todo: tell charisma that the outport is created
    }
};

template <class d>
class CkOutPort: public CharismaOutPort
{
  public:
    CkOutPort(const char *name) { _create(name); }
    void emit(d &_d)
    {
      emit((void *) &_d, sizeof(d));
    }
};

template <class d>
class CkInPort : public CharismaInPort
{
  private:
    CkCallback cb;
    CkInPort() {} // prevent illegal inports
  public:
    CkInPort(const char *name, int ep,const CkChareID &id)
    {
      _create(name);
      CkCallback _cb(ep,id);
      cb = _cb;
    }
    CkInPort(const char *name, int ep,int onPE,const CkGroupID &id)
    {
      _create(name);
      CkCallback _cb(ep,onPE,id);
      cb = _cb;
    }
    CkInPort(const char *name, int ep,const CkArrayIndex &idx,
             const CkArrayID &id)
    {
      _create(name);
      CkCallback _cb(ep,idx,id);
      cb = _cb;
    }
    void send(void *data, int len)
    {
      send((d&) (*((d*)data)));
    }
    void send(d &_d)
    {
      int impl_off=0,impl_arrstart=0;
      { //Find the size of the PUP'd data
        PUP::sizer implP;
        implP|_d;
        impl_arrstart=CK_ALIGN(implP.size(),16);
        impl_off+=impl_arrstart;
      }
      CkMarshallMsg *impl_msg=new (impl_off,0)CkMarshallMsg;
      { //Copy over the PUP'd data
        PUP::toMem implP((void *)impl_msg->msgBuf);
        implP|_d;
      }
      CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
      impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
      cb.send(impl_msg);
    }
};

class CkOutPortString : public CharismaOutPort
{
  public:
    CkOutPortString(const char *name) { _create(name); }
    void emit(char *str)
    {
      CharismaOutPort::emit((void *) str, (int) strlen(str)+1);
    }
};

class CkInPortString : public CharismaInPort
{
  private:
    CkCallback cb;
    CkInPortString() {} // prevent illegal inports
  public:
    CkInPortString(const char *name, int ep,const CkChareID &id)
    {
      _create(name);
      CkCallback _cb(ep,id);
      cb = _cb;
    }
    CkInPortString(const char *name, int ep,int onPE,const CkGroupID &id)
    {
      _create(name);
      CkCallback _cb(ep,onPE,id);
      cb = _cb;
    }
    CkInPortString(const char *name, int ep,const CkArrayIndex &idx,
                   const CkArrayID &id)
    {
      _create(name);
      CkCallback _cb(ep,idx,id);
      cb = _cb;
    }
    void send(void *data, int len)
    {
      send((char *) data);
    }
    void send(char *str)
    {
      CkMarshallMsg *impl_msg=new (strlen(str)+1,0)CkMarshallMsg;
      strcpy((char *)(impl_msg->msgBuf),str);
      CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
      impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
      cb.send(impl_msg);
    }
};

template <class d>
class CkOutPortArray : public CharismaOutPort
{
  public:
    CkOutPortArray(const char *name) { _create(name); }
    void emit(int n, const d *a)
    {
      emit((void *)a, n*sizeof(d));
    }
};

template <class d>
class CkInPortArray : public CharismaInPort
{
  private:
    CkCallback cb;
    CkInPortArray() {} // prevent illegal inports
  public:
    CkInPortArray(const char *name, int ep,const CkChareID &id)
    {
      _create(name);
      CkCallback _cb(ep,id);
      cb = _cb;
    }
    CkInPortArray(const char *name, int ep,int onPE,const CkGroupID &id)
    {
      _create(name);
      CkCallback _cb(ep,onPE,id);
      cb = _cb;
    }
    CkInPortArray(const char *name, int ep,const CkArrayIndex &idx,
                  const CkArrayID &id)
    {
      _create(name);
      CkCallback _cb(ep,idx,id);
      cb = _cb;
    }
    void send(void *data, int len)
    {
      send(len/sizeof(d), (const d*) data);
    }
    void send(int n, const d *a)
    {
      //Marshall: int n, const int *a
      int impl_off=0,impl_arrstart=0;
      int impl_off_a, impl_cnt_a;
      impl_off_a=impl_off=CK_ALIGN(impl_off,sizeof(int));
      impl_off+=(impl_cnt_a=sizeof(int)*(n));
      { //Find the size of the PUP'd data
        PUP::sizer implP;
        implP|n;
        implP|impl_off_a;
        impl_arrstart=CK_ALIGN(implP.size(),16);
        impl_off+=impl_arrstart;
      }
      CkMarshallMsg *impl_msg=new (impl_off,0)CkMarshallMsg;
      { //Copy over the PUP'd data
        PUP::toMem implP((void *)impl_msg->msgBuf);
        implP|n;
        implP|impl_off_a;
      }
      char *impl_buf=impl_msg->msgBuf+impl_arrstart;
      memcpy(impl_buf+impl_off_a,a,impl_cnt_a);
    }
};

class CkOutPortVoid : public CharismaOutPort
{
  public:
    CkOutPortVoid(const char *name) { _create(name); }
    void emit(void)
    {
      CharismaOutPort::emit((void*) 0, 0);
    }
};

class CkInPortVoid : public CharismaInPort
{
  private:
    CkCallback cb;
    CkInPortVoid() {} // prevent illegal inports
  public:
    CkInPortVoid(const char *name, int ep,const CkChareID &id)
    {
      _create(name);
      CkCallback _cb(ep,id);
      cb = _cb;
    }
    CkInPortVoid(const char *name, int ep,int onPE,const CkGroupID &id)
    {
      _create(name);
      CkCallback _cb(ep,onPE,id);
      cb = _cb;
    }
    CkInPortVoid(const char *name, int ep,const CkArrayIndex &idx,
                 const CkArrayID &id)
    {
      _create(name);
      CkCallback _cb(ep,idx,id);
      cb = _cb;
    }
    void send(void *data, int len)
    {
      send();
    }
    void send(void)
    {
      void *impl_msg = CkAllocSysMsg();
      cb.send(impl_msg);
    }
};

template <class d>
class CkOutPortMsg : public CharismaOutPort
{
  public:
    CkOutPortMsg(const char *name) { _create(name); }
    void emit(d *data)
    {
      // TODO: do message packing, and get the actual buffer size
      emit((void*) data, sizeof(d));
    }
};

template <class d>
class CkInPortMsg : public CharismaInPort
{
  private:
    CkCallback cb;
    CkInPortMsg() {} // prevent illegal inports
  public:
    CkInPortMsg(const char *name, int ep,const CkChareID &id)
    {
      _create(name);
      CkCallback _cb(ep,id);
      cb = _cb;
    }
    CkInPortMsg(const char *name, int ep,int onPE,const CkGroupID &id)
    {
      _create(name);
      CkCallback _cb(ep,onPE,id);
      cb = _cb;
    }
    CkInPortMsg(const char *name, int ep,const CkArrayIndex &idx,
                const CkArrayID &id)
    {
      _create(name);
      CkCallback _cb(ep,idx,id);
      cb = _cb;
    }
    void send(void *data, int len)
    {
      send((d*) _d);
    }
    void send(d *_d)
    {
      cb.send(_d);
    }
};

class Charisma : public Group
{
  public:
    Charisma(void) { }
};

#endif
