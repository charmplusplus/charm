#ifndef LBCOMM_H
#define LBCOMM_H

#include "converse.h"
#include "lbdb.h"

class LBCommData {

friend class LBCommTable;

public:
  LBCommData(int _src_proc, LDOMid _destOM, LDObjid _destObj) {
    from_proc = CmiTrue;
    src_proc = _src_proc;
    destOM = _destOM;
    destObj = _destObj;
    n_messages = 0;
    n_bytes = 0;
    mykey = compute_key();
  };

  LBCommData(LDObjHandle _srcObj, LDOMid _destOM, LDObjid _destObj) {
    from_proc = CmiFalse;
    srcObj = _srcObj;
    destOM = _destOM;
    destObj = _destObj;
    n_messages = 0;
    n_bytes = 0;
    mykey = compute_key();
  };

  LBCommData(const LBCommData& d) {
    if (from_proc = d.from_proc)
      src_proc = d.src_proc;
    else srcObj = d.srcObj;
    destOM = d.destOM;
    destObj = d.destObj;
    n_messages = d.n_messages;
    n_bytes = d.n_messages;
    mykey = d.mykey;
  };

  ~LBCommData() { };

  LBCommData& operator=(const LBCommData& d) {
    if (from_proc = d.from_proc)
      src_proc = d.src_proc;
    else srcObj = d.srcObj;
    destOM = d.destOM;
    destObj = d.destObj;
    n_messages = d.n_messages;
    n_bytes = d.n_messages;
    mykey = d.mykey;
    return *this;
  };

  void addMessage(int bytes) {
    n_messages++;
    n_bytes += bytes;
  };

  int key() const { return mykey; };
  CmiBool equal(const LBCommData _d2) const;

private:
  LBCommData() {};
  
  int compute_key();
  int hash(int i, int m) const;

  int mykey;
  CmiBool from_proc;
  int src_proc;
  LDObjHandle srcObj;
  LDOMid destOM;
  LDObjid destObj;
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

  LBCommData* HashInsert(const LBCommData data);
  LBCommData* HashInsertUnique(const LBCommData data);
  LBCommData* HashSearch(const LBCommData data);
  int CommCount() { return in_use; };
  void GetCommData(LDCommData* data);
	
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

  enum { initial_sz = 65536 };
  enum TableState { nil, InUse } ;
  LBCommData* set;
  TableState* state;
  int cur_sz;
  int in_use;
};


#endif
