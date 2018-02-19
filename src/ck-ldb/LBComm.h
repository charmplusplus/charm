/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBCOMM_H
#define LBCOMM_H

#include "converse.h"
#include "lbdb.h"

class LBObj; //Forward declaration
template <class T> class CkVec; //Forward declaration

// point to point communication data
class LBCommData {

friend class LBCommTable;

public:
  LBCommData(int _src_proc, LDOMid _destOM, CmiUInt8 _destObj, int _destObjProc) {
    src_proc = _src_proc;
    destObj.init_objmsg(_destOM, _destObj, _destObjProc);
    n_messages = 0;
    n_bytes = 0;
    mykey = compute_key();
  };

  LBCommData(LDObjHandle _srcObj, LDOMid _destOM, CmiUInt8 _destObj, int _destObjProc) {
    src_proc = -1;
    srcObj = _srcObj;
    destObj.init_objmsg(_destOM, _destObj, _destObjProc);
    n_messages = 0;
    n_bytes = 0;
    mykey = compute_key();
  };

  // multicast
  LBCommData(LDObjHandle _srcObj, LDOMid _destOM, CmiUInt8 *_destObjs, int _nobjs) {
    src_proc = -1;
    srcObj = _srcObj;
    destObj.init_mcastmsg(_destOM, _destObjs, _nobjs);
    n_messages = 0;
    n_bytes = 0;
    mykey = compute_key();
  };

  LBCommData(const LBCommData& d) {
    src_proc = d.src_proc;
    if (!from_proc()) {
      srcObj = d.srcObj;
//      srcOM = d.srcOM;
    }
    destObj = d.destObj;
    n_messages = d.n_messages;
    n_bytes = d.n_bytes;
    mykey = d.mykey;
  };

  ~LBCommData() { };

  LBCommData& operator=(const LBCommData& d) {
    src_proc = d.src_proc;
    if (!from_proc()) { 
      srcObj = d.srcObj;
//      srcOM = d.srcOM;
    }
    destObj = d.destObj;
    n_messages = d.n_messages;
    n_bytes = d.n_bytes;
    mykey = d.mykey;
    return *this;
  };

  void addMessage(int bytes, int nMsgs=1) {
    n_messages += nMsgs;
    n_bytes += bytes;
  };

  inline int key() const { return mykey; };
  bool equal(const LBCommData &_d2) const;

  inline int from_proc() const { return (src_proc != -1); }
private:
  LBCommData(): mykey(0), src_proc(0), n_messages(0), n_bytes(0) {};
  
  int compute_key();
  int hash(const int i, const int m) const;

  int mykey;
  int src_proc;
  LDObjHandle srcObj;
  LDCommDesc   destObj;
  int n_messages;
  int n_bytes;
};

class LBCommTable {
public:

  LBCommTable() {
    NewTable(initial_sz);
  };

  ~LBCommTable() {
    delete [] set;
    delete [] state;
  };

  LBCommData* HashInsert(const LBCommData &data);
  LBCommData* HashInsertUnique(const LBCommData &data);
  LBCommData* HashSearch(const LBCommData &data);
  int CommCount() { return in_use; };
  void GetCommData(LDCommData* data);
  void GetCommInfo(int& bytes, int& msgs, int& withinpebytes,
      int& outsidepebytes, int& num_nghbor, int& hops, int& hopbytes);
	
private:
  void NewTable(int _sz) {
    set = new LBCommData[_sz];
    state = new TableState[_sz];
    cur_sz = _sz;
    in_use = 0;
    for(int i=0; i < _sz; i++)
      state[i] = nil;
  };
  
  void Resize();

#ifdef __BIGSIM__
  enum { initial_sz = 1 };
#else
  enum { initial_sz = 10000 };
#endif
  enum TableState : uint8_t { nil, InUse } ;
  LBCommData* set;
  TableState* state;
  int cur_sz;
  int in_use;
public:
  int useMem() { return cur_sz*(sizeof(LBCommData) + sizeof(TableState)) + sizeof(LBCommTable); }
};


#endif

/*@}*/
