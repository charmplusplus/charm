#ifndef _DECL_refine_H_
#define _DECL_refine_H_
#include "charm++.h"
/* DECLS: message chunkMsg;
 */
class chunkMsg;
class CMessage_chunkMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_chunkMsg() {};
    static void *pack(chunkMsg *p);
    static chunkMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message nodeMsg;
 */
class nodeMsg;
class CMessage_nodeMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_nodeMsg() {};
    static void *pack(nodeMsg *p);
    static nodeMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message edgeMsg;
 */
class edgeMsg;
class CMessage_edgeMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_edgeMsg() {};
    static void *pack(edgeMsg *p);
    static edgeMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message remoteEdgeMsg;
 */
class remoteEdgeMsg;
class CMessage_remoteEdgeMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_remoteEdgeMsg() {};
    static void *pack(remoteEdgeMsg *p);
    static remoteEdgeMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message elementMsg;
 */
class elementMsg;
class CMessage_elementMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_elementMsg() {};
    static void *pack(elementMsg *p);
    static elementMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message femElementMsg;
 */
class femElementMsg;
class CMessage_femElementMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_femElementMsg() {};
    static void *pack(femElementMsg *p);
    static femElementMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message ghostElementMsg;
 */
class ghostElementMsg;
class CMessage_ghostElementMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_ghostElementMsg() {};
    static void *pack(ghostElementMsg *p);
    static ghostElementMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message refineMsg;
 */
class refineMsg;
class CMessage_refineMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_refineMsg() {};
    static void *pack(refineMsg *p);
    static refineMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message coarsenMsg;
 */
class coarsenMsg;
class CMessage_coarsenMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_coarsenMsg() {};
    static void *pack(coarsenMsg *p);
    static coarsenMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message collapseMsg;
 */
class collapseMsg;
class CMessage_collapseMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_collapseMsg() {};
    static void *pack(collapseMsg *p);
    static collapseMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message splitInMsg;
 */
class splitInMsg;
class CMessage_splitInMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_splitInMsg() {};
    static void *pack(splitInMsg *p);
    static splitInMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message splitOutMsg;
 */
class splitOutMsg;
class CMessage_splitOutMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_splitOutMsg() {};
    static void *pack(splitOutMsg *p);
    static splitOutMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message updateMsg;
 */
class updateMsg;
class CMessage_updateMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_updateMsg() {};
    static void *pack(updateMsg *p);
    static updateMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message specialRequestMsg;
 */
class specialRequestMsg;
class CMessage_specialRequestMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_specialRequestMsg() {};
    static void *pack(specialRequestMsg *p);
    static specialRequestMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message specialResponseMsg;
 */
class specialResponseMsg;
class CMessage_specialResponseMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_specialResponseMsg() {};
    static void *pack(specialResponseMsg *p);
    static specialResponseMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message refMsg;
 */
class refMsg;
class CMessage_refMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_refMsg() {};
    static void *pack(refMsg *p);
    static refMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message drefMsg;
 */
class drefMsg;
class CMessage_drefMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_drefMsg() {};
    static void *pack(drefMsg *p);
    static drefMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message edgeUpdateMsg;
 */
class edgeUpdateMsg;
class CMessage_edgeUpdateMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_edgeUpdateMsg() {};
    static void *pack(edgeUpdateMsg *p);
    static edgeUpdateMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message intMsg;
 */
class intMsg;
class CMessage_intMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_intMsg() {};
    static void *pack(intMsg *p);
    static intMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message doubleMsg;
 */
class doubleMsg;
class CMessage_doubleMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_doubleMsg() {};
    static void *pack(doubleMsg *p);
    static doubleMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: readonly CProxy_chunk mesh;
 */


/* DECLS: array chunk: ArrayElement{
chunk(CkMigrateMessage* impl_msg);
chunk(chunkMsg* impl_msg);
void addRemoteEdge(remoteEdgeMsg* impl_msg);
void refineElement(refineMsg* impl_msg);
threaded void refiningElements(void);
void coarsenElement(coarsenMsg* impl_msg);
threaded void coarseningElements(void);
sync nodeMsg* getNode(intMsg* impl_msg);
sync refMsg* getEdge(collapseMsg* impl_msg);
sync void setBorder(intMsg* impl_msg);
sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
sync splitOutMsg* split(splitInMsg* impl_msg);
sync void collapseHelp(collapseMsg* impl_msg);
void checkPending(refMsg* impl_msg);
void checkPending(drefMsg* impl_msg);
sync void updateElement(updateMsg* impl_msg);
sync void updateElementEdge(updateMsg* impl_msg);
void updateReferences(updateMsg* impl_msg);
threaded sync doubleMsg* getArea(intMsg* impl_msg);
threaded sync nodeMsg* midpoint(intMsg* impl_msg);
sync intMsg* setPending(intMsg* impl_msg);
sync void unsetPending(intMsg* impl_msg);
sync intMsg* isPending(intMsg* impl_msg);
sync intMsg* lockNode(intMsg* impl_msg);
sync void unlockNode(intMsg* impl_msg);
threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
sync refMsg* getNeighbor(refMsg* impl_msg);
sync refMsg* getNotNode(refMsg* impl_msg);
sync refMsg* getNotElem(refMsg* impl_msg);
sync void setTargetArea(doubleMsg* impl_msg);
sync void resetTargetArea(doubleMsg* impl_msg);
sync void updateEdges(edgeUpdateMsg* impl_msg);
sync void updateNodeCoords(nodeMsg* impl_msg);
void reportPos(nodeMsg* impl_msg);
threaded sync void print(void);
threaded sync void out_print(void);
void freshen(void);
void deriveBorderNodes(void);
void tweakMesh(void);
void improveChunk(void);
threaded sync void improve(void);
void addNode(nodeMsg* impl_msg);
void addEdge(edgeMsg* impl_msg);
void addElement(elementMsg* impl_msg);
sync void removeNode(intMsg* impl_msg);
sync void removeEdge(intMsg* impl_msg);
sync void removeElement(intMsg* impl_msg);
sync void updateNode(updateMsg* impl_msg);
sync refMsg* getOpposingNode(refMsg* impl_msg);
};
 */
 class chunk;
 class CkIndex_chunk;
 class CProxy_chunk;
 class CProxyElement_chunk;
 class CProxySection_chunk;
/* --------------- index object ------------------ */
class CkIndex_chunk{
  public:
    typedef chunk local_t;
    typedef CkIndex_chunk index_t;
    typedef CProxy_chunk proxy_t;
    typedef CProxyElement_chunk element_t;
    typedef CProxySection_chunk section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: chunk(CkMigrateMessage* impl_msg);
 */
    static int __idx_chunk_CkMigrateMessage;
    static int ckNew(CkMigrateMessage* impl_msg) { return __idx_chunk_CkMigrateMessage; }
    static void _call_chunk_CkMigrateMessage(void* impl_msg,chunk* impl_obj);

/* DECLS: chunk(chunkMsg* impl_msg);
 */
    static int __idx_chunk_chunkMsg;
    static int ckNew(chunkMsg* impl_msg) { return __idx_chunk_chunkMsg; }
    static void _call_chunk_chunkMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
    static int __idx_addRemoteEdge_remoteEdgeMsg;
    static int addRemoteEdge(remoteEdgeMsg* impl_msg) { return __idx_addRemoteEdge_remoteEdgeMsg; }
    static void _call_addRemoteEdge_remoteEdgeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void refineElement(refineMsg* impl_msg);
 */
    static int __idx_refineElement_refineMsg;
    static int refineElement(refineMsg* impl_msg) { return __idx_refineElement_refineMsg; }
    static void _call_refineElement_refineMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded void refiningElements(void);
 */
    static int __idx_refiningElements_void;
    static int refiningElements(void) { return __idx_refiningElements_void; }
    static void _call_refiningElements_void(void* impl_msg,chunk* impl_obj);
    static void _callthr_refiningElements_void(CkThrCallArg *);

/* DECLS: void coarsenElement(coarsenMsg* impl_msg);
 */
    static int __idx_coarsenElement_coarsenMsg;
    static int coarsenElement(coarsenMsg* impl_msg) { return __idx_coarsenElement_coarsenMsg; }
    static void _call_coarsenElement_coarsenMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded void coarseningElements(void);
 */
    static int __idx_coarseningElements_void;
    static int coarseningElements(void) { return __idx_coarseningElements_void; }
    static void _call_coarseningElements_void(void* impl_msg,chunk* impl_obj);
    static void _callthr_coarseningElements_void(CkThrCallArg *);

/* DECLS: sync nodeMsg* getNode(intMsg* impl_msg);
 */
    static int __idx_getNode_intMsg;
    static int getNode(intMsg* impl_msg) { return __idx_getNode_intMsg; }
    static void _call_getNode_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */
    static int __idx_getEdge_collapseMsg;
    static int getEdge(collapseMsg* impl_msg) { return __idx_getEdge_collapseMsg; }
    static void _call_getEdge_collapseMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void setBorder(intMsg* impl_msg);
 */
    static int __idx_setBorder_intMsg;
    static int setBorder(intMsg* impl_msg) { return __idx_setBorder_intMsg; }
    static void _call_setBorder_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */
    static int __idx_safeToMoveNode_nodeMsg;
    static int safeToMoveNode(nodeMsg* impl_msg) { return __idx_safeToMoveNode_nodeMsg; }
    static void _call_safeToMoveNode_nodeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */
    static int __idx_split_splitInMsg;
    static int split(splitInMsg* impl_msg) { return __idx_split_splitInMsg; }
    static void _call_split_splitInMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void collapseHelp(collapseMsg* impl_msg);
 */
    static int __idx_collapseHelp_collapseMsg;
    static int collapseHelp(collapseMsg* impl_msg) { return __idx_collapseHelp_collapseMsg; }
    static void _call_collapseHelp_collapseMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void checkPending(refMsg* impl_msg);
 */
    static int __idx_checkPending_refMsg;
    static int checkPending(refMsg* impl_msg) { return __idx_checkPending_refMsg; }
    static void _call_checkPending_refMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void checkPending(drefMsg* impl_msg);
 */
    static int __idx_checkPending_drefMsg;
    static int checkPending(drefMsg* impl_msg) { return __idx_checkPending_drefMsg; }
    static void _call_checkPending_drefMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void updateElement(updateMsg* impl_msg);
 */
    static int __idx_updateElement_updateMsg;
    static int updateElement(updateMsg* impl_msg) { return __idx_updateElement_updateMsg; }
    static void _call_updateElement_updateMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void updateElementEdge(updateMsg* impl_msg);
 */
    static int __idx_updateElementEdge_updateMsg;
    static int updateElementEdge(updateMsg* impl_msg) { return __idx_updateElementEdge_updateMsg; }
    static void _call_updateElementEdge_updateMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void updateReferences(updateMsg* impl_msg);
 */
    static int __idx_updateReferences_updateMsg;
    static int updateReferences(updateMsg* impl_msg) { return __idx_updateReferences_updateMsg; }
    static void _call_updateReferences_updateMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */
    static int __idx_getArea_intMsg;
    static int getArea(intMsg* impl_msg) { return __idx_getArea_intMsg; }
    static void _call_getArea_intMsg(void* impl_msg,chunk* impl_obj);
    static void _callthr_getArea_intMsg(CkThrCallArg *);

/* DECLS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */
    static int __idx_midpoint_intMsg;
    static int midpoint(intMsg* impl_msg) { return __idx_midpoint_intMsg; }
    static void _call_midpoint_intMsg(void* impl_msg,chunk* impl_obj);
    static void _callthr_midpoint_intMsg(CkThrCallArg *);

/* DECLS: sync intMsg* setPending(intMsg* impl_msg);
 */
    static int __idx_setPending_intMsg;
    static int setPending(intMsg* impl_msg) { return __idx_setPending_intMsg; }
    static void _call_setPending_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void unsetPending(intMsg* impl_msg);
 */
    static int __idx_unsetPending_intMsg;
    static int unsetPending(intMsg* impl_msg) { return __idx_unsetPending_intMsg; }
    static void _call_unsetPending_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync intMsg* isPending(intMsg* impl_msg);
 */
    static int __idx_isPending_intMsg;
    static int isPending(intMsg* impl_msg) { return __idx_isPending_intMsg; }
    static void _call_isPending_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync intMsg* lockNode(intMsg* impl_msg);
 */
    static int __idx_lockNode_intMsg;
    static int lockNode(intMsg* impl_msg) { return __idx_lockNode_intMsg; }
    static void _call_lockNode_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void unlockNode(intMsg* impl_msg);
 */
    static int __idx_unlockNode_intMsg;
    static int unlockNode(intMsg* impl_msg) { return __idx_unlockNode_intMsg; }
    static void _call_unlockNode_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */
    static int __idx_isLongestEdge_refMsg;
    static int isLongestEdge(refMsg* impl_msg) { return __idx_isLongestEdge_refMsg; }
    static void _call_isLongestEdge_refMsg(void* impl_msg,chunk* impl_obj);
    static void _callthr_isLongestEdge_refMsg(CkThrCallArg *);

/* DECLS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */
    static int __idx_getNeighbor_refMsg;
    static int getNeighbor(refMsg* impl_msg) { return __idx_getNeighbor_refMsg; }
    static void _call_getNeighbor_refMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync refMsg* getNotNode(refMsg* impl_msg);
 */
    static int __idx_getNotNode_refMsg;
    static int getNotNode(refMsg* impl_msg) { return __idx_getNotNode_refMsg; }
    static void _call_getNotNode_refMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync refMsg* getNotElem(refMsg* impl_msg);
 */
    static int __idx_getNotElem_refMsg;
    static int getNotElem(refMsg* impl_msg) { return __idx_getNotElem_refMsg; }
    static void _call_getNotElem_refMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void setTargetArea(doubleMsg* impl_msg);
 */
    static int __idx_setTargetArea_doubleMsg;
    static int setTargetArea(doubleMsg* impl_msg) { return __idx_setTargetArea_doubleMsg; }
    static void _call_setTargetArea_doubleMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void resetTargetArea(doubleMsg* impl_msg);
 */
    static int __idx_resetTargetArea_doubleMsg;
    static int resetTargetArea(doubleMsg* impl_msg) { return __idx_resetTargetArea_doubleMsg; }
    static void _call_resetTargetArea_doubleMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */
    static int __idx_updateEdges_edgeUpdateMsg;
    static int updateEdges(edgeUpdateMsg* impl_msg) { return __idx_updateEdges_edgeUpdateMsg; }
    static void _call_updateEdges_edgeUpdateMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */
    static int __idx_updateNodeCoords_nodeMsg;
    static int updateNodeCoords(nodeMsg* impl_msg) { return __idx_updateNodeCoords_nodeMsg; }
    static void _call_updateNodeCoords_nodeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void reportPos(nodeMsg* impl_msg);
 */
    static int __idx_reportPos_nodeMsg;
    static int reportPos(nodeMsg* impl_msg) { return __idx_reportPos_nodeMsg; }
    static void _call_reportPos_nodeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded sync void print(void);
 */
    static int __idx_print_void;
    static int print(void) { return __idx_print_void; }
    static void _call_print_void(void* impl_msg,chunk* impl_obj);
    static void _callthr_print_void(CkThrCallArg *);

/* DECLS: threaded sync void out_print(void);
 */
    static int __idx_out_print_void;
    static int out_print(void) { return __idx_out_print_void; }
    static void _call_out_print_void(void* impl_msg,chunk* impl_obj);
    static void _callthr_out_print_void(CkThrCallArg *);

/* DECLS: void freshen(void);
 */
    static int __idx_freshen_void;
    static int freshen(void) { return __idx_freshen_void; }
    static void _call_freshen_void(void* impl_msg,chunk* impl_obj);

/* DECLS: void deriveBorderNodes(void);
 */
    static int __idx_deriveBorderNodes_void;
    static int deriveBorderNodes(void) { return __idx_deriveBorderNodes_void; }
    static void _call_deriveBorderNodes_void(void* impl_msg,chunk* impl_obj);

/* DECLS: void tweakMesh(void);
 */
    static int __idx_tweakMesh_void;
    static int tweakMesh(void) { return __idx_tweakMesh_void; }
    static void _call_tweakMesh_void(void* impl_msg,chunk* impl_obj);

/* DECLS: void improveChunk(void);
 */
    static int __idx_improveChunk_void;
    static int improveChunk(void) { return __idx_improveChunk_void; }
    static void _call_improveChunk_void(void* impl_msg,chunk* impl_obj);

/* DECLS: threaded sync void improve(void);
 */
    static int __idx_improve_void;
    static int improve(void) { return __idx_improve_void; }
    static void _call_improve_void(void* impl_msg,chunk* impl_obj);
    static void _callthr_improve_void(CkThrCallArg *);

/* DECLS: void addNode(nodeMsg* impl_msg);
 */
    static int __idx_addNode_nodeMsg;
    static int addNode(nodeMsg* impl_msg) { return __idx_addNode_nodeMsg; }
    static void _call_addNode_nodeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void addEdge(edgeMsg* impl_msg);
 */
    static int __idx_addEdge_edgeMsg;
    static int addEdge(edgeMsg* impl_msg) { return __idx_addEdge_edgeMsg; }
    static void _call_addEdge_edgeMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: void addElement(elementMsg* impl_msg);
 */
    static int __idx_addElement_elementMsg;
    static int addElement(elementMsg* impl_msg) { return __idx_addElement_elementMsg; }
    static void _call_addElement_elementMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void removeNode(intMsg* impl_msg);
 */
    static int __idx_removeNode_intMsg;
    static int removeNode(intMsg* impl_msg) { return __idx_removeNode_intMsg; }
    static void _call_removeNode_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void removeEdge(intMsg* impl_msg);
 */
    static int __idx_removeEdge_intMsg;
    static int removeEdge(intMsg* impl_msg) { return __idx_removeEdge_intMsg; }
    static void _call_removeEdge_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void removeElement(intMsg* impl_msg);
 */
    static int __idx_removeElement_intMsg;
    static int removeElement(intMsg* impl_msg) { return __idx_removeElement_intMsg; }
    static void _call_removeElement_intMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync void updateNode(updateMsg* impl_msg);
 */
    static int __idx_updateNode_updateMsg;
    static int updateNode(updateMsg* impl_msg) { return __idx_updateNode_updateMsg; }
    static void _call_updateNode_updateMsg(void* impl_msg,chunk* impl_obj);

/* DECLS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
    static int __idx_getOpposingNode_refMsg;
    static int getOpposingNode(refMsg* impl_msg) { return __idx_getOpposingNode_refMsg; }
    static void _call_getOpposingNode_refMsg(void* impl_msg,chunk* impl_obj);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_chunk : public CProxyElement_ArrayElement{
  public:
    typedef chunk local_t;
    typedef CkIndex_chunk index_t;
    typedef CProxy_chunk proxy_t;
    typedef CProxyElement_chunk element_t;
    typedef CProxySection_chunk section_t;

    CProxyElement_chunk(void) {}
    CProxyElement_chunk(const ArrayElement *e) : CProxyElement_ArrayElement(e){  }
    void ckDelegate(CkDelegateMgr *to) {
      CProxyElement_ArrayElement::ckDelegate(to);
    }
    void ckUndelegate(void) {
      CProxyElement_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxyElement_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_ELEMENT(CProxyElement_ArrayElement)
    chunk *ckLocal(void) const
      { return (chunk *)CProxyElement_ArrayElement::ckLocal(); }
    CProxyElement_chunk(const CkArrayID &aid,const CkArrayIndex1D &idx,CkGroupID dTo)
        :CProxyElement_ArrayElement(aid,idx,dTo) {}
    CProxyElement_chunk(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_ArrayElement(aid,idx) {}
/* DECLS: chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: chunk(chunkMsg* impl_msg);
 */
    void insert(chunkMsg* impl_msg, int onPE=-1);
/* DECLS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
    void addRemoteEdge(remoteEdgeMsg* impl_msg) ;

/* DECLS: void refineElement(refineMsg* impl_msg);
 */
    void refineElement(refineMsg* impl_msg) ;

/* DECLS: threaded void refiningElements(void);
 */
    void refiningElements(void) ;

/* DECLS: void coarsenElement(coarsenMsg* impl_msg);
 */
    void coarsenElement(coarsenMsg* impl_msg) ;

/* DECLS: threaded void coarseningElements(void);
 */
    void coarseningElements(void) ;

/* DECLS: sync nodeMsg* getNode(intMsg* impl_msg);
 */
    nodeMsg* getNode(intMsg* impl_msg) ;

/* DECLS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */
    refMsg* getEdge(collapseMsg* impl_msg) ;

/* DECLS: sync void setBorder(intMsg* impl_msg);
 */
    void setBorder(intMsg* impl_msg) ;

/* DECLS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */
    intMsg* safeToMoveNode(nodeMsg* impl_msg) ;

/* DECLS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */
    splitOutMsg* split(splitInMsg* impl_msg) ;

/* DECLS: sync void collapseHelp(collapseMsg* impl_msg);
 */
    void collapseHelp(collapseMsg* impl_msg) ;

/* DECLS: void checkPending(refMsg* impl_msg);
 */
    void checkPending(refMsg* impl_msg) ;

/* DECLS: void checkPending(drefMsg* impl_msg);
 */
    void checkPending(drefMsg* impl_msg) ;

/* DECLS: sync void updateElement(updateMsg* impl_msg);
 */
    void updateElement(updateMsg* impl_msg) ;

/* DECLS: sync void updateElementEdge(updateMsg* impl_msg);
 */
    void updateElementEdge(updateMsg* impl_msg) ;

/* DECLS: void updateReferences(updateMsg* impl_msg);
 */
    void updateReferences(updateMsg* impl_msg) ;

/* DECLS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */
    doubleMsg* getArea(intMsg* impl_msg) ;

/* DECLS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */
    nodeMsg* midpoint(intMsg* impl_msg) ;

/* DECLS: sync intMsg* setPending(intMsg* impl_msg);
 */
    intMsg* setPending(intMsg* impl_msg) ;

/* DECLS: sync void unsetPending(intMsg* impl_msg);
 */
    void unsetPending(intMsg* impl_msg) ;

/* DECLS: sync intMsg* isPending(intMsg* impl_msg);
 */
    intMsg* isPending(intMsg* impl_msg) ;

/* DECLS: sync intMsg* lockNode(intMsg* impl_msg);
 */
    intMsg* lockNode(intMsg* impl_msg) ;

/* DECLS: sync void unlockNode(intMsg* impl_msg);
 */
    void unlockNode(intMsg* impl_msg) ;

/* DECLS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */
    intMsg* isLongestEdge(refMsg* impl_msg) ;

/* DECLS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */
    refMsg* getNeighbor(refMsg* impl_msg) ;

/* DECLS: sync refMsg* getNotNode(refMsg* impl_msg);
 */
    refMsg* getNotNode(refMsg* impl_msg) ;

/* DECLS: sync refMsg* getNotElem(refMsg* impl_msg);
 */
    refMsg* getNotElem(refMsg* impl_msg) ;

/* DECLS: sync void setTargetArea(doubleMsg* impl_msg);
 */
    void setTargetArea(doubleMsg* impl_msg) ;

/* DECLS: sync void resetTargetArea(doubleMsg* impl_msg);
 */
    void resetTargetArea(doubleMsg* impl_msg) ;

/* DECLS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */
    void updateEdges(edgeUpdateMsg* impl_msg) ;

/* DECLS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */
    void updateNodeCoords(nodeMsg* impl_msg) ;

/* DECLS: void reportPos(nodeMsg* impl_msg);
 */
    void reportPos(nodeMsg* impl_msg) ;

/* DECLS: threaded sync void print(void);
 */
    void print(void) ;

/* DECLS: threaded sync void out_print(void);
 */
    void out_print(void) ;

/* DECLS: void freshen(void);
 */
    void freshen(void) ;

/* DECLS: void deriveBorderNodes(void);
 */
    void deriveBorderNodes(void) ;

/* DECLS: void tweakMesh(void);
 */
    void tweakMesh(void) ;

/* DECLS: void improveChunk(void);
 */
    void improveChunk(void) ;

/* DECLS: threaded sync void improve(void);
 */
    void improve(void) ;

/* DECLS: void addNode(nodeMsg* impl_msg);
 */
    void addNode(nodeMsg* impl_msg) ;

/* DECLS: void addEdge(edgeMsg* impl_msg);
 */
    void addEdge(edgeMsg* impl_msg) ;

/* DECLS: void addElement(elementMsg* impl_msg);
 */
    void addElement(elementMsg* impl_msg) ;

/* DECLS: sync void removeNode(intMsg* impl_msg);
 */
    void removeNode(intMsg* impl_msg) ;

/* DECLS: sync void removeEdge(intMsg* impl_msg);
 */
    void removeEdge(intMsg* impl_msg) ;

/* DECLS: sync void removeElement(intMsg* impl_msg);
 */
    void removeElement(intMsg* impl_msg) ;

/* DECLS: sync void updateNode(updateMsg* impl_msg);
 */
    void updateNode(updateMsg* impl_msg) ;

/* DECLS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
    refMsg* getOpposingNode(refMsg* impl_msg) ;
};
PUPmarshall(CProxyElement_chunk);
/* ---------------- collective proxy -------------- */
 class CProxy_chunk : public CProxy_ArrayElement{
  public:
    typedef chunk local_t;
    typedef CkIndex_chunk index_t;
    typedef CProxy_chunk proxy_t;
    typedef CProxyElement_chunk element_t;
    typedef CProxySection_chunk section_t;

    CProxy_chunk(void) {}
    CProxy_chunk(const ArrayElement *e) : CProxy_ArrayElement(e){  }
    void ckDelegate(CkDelegateMgr *to) {
      CProxy_ArrayElement::ckDelegate(to);
    }
    void ckUndelegate(void) {
      CProxy_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxy_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY(CProxy_ArrayElement)
    static CkArrayID ckNew(void) {return ckCreateEmptyArray();}
//Generalized array indexing:
    CProxyElement_chunk operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_chunk operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_chunk operator [] (int idx) const 
        {return CProxyElement_chunk(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}
    CProxyElement_chunk operator () (int idx) const 
        {return CProxyElement_chunk(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}
    CProxy_chunk(const CkArrayID &aid,CkGroupID dTo) 
        :CProxy_ArrayElement(aid,dTo) {}
    CProxy_chunk(const CkArrayID &aid) 
        :CProxy_ArrayElement(aid) {}
/* DECLS: chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: chunk(chunkMsg* impl_msg);
 */
    static CkArrayID ckNew(chunkMsg* impl_msg, const CkArrayOptions &opts);

/* DECLS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
    void addRemoteEdge(remoteEdgeMsg* impl_msg) ;

/* DECLS: void refineElement(refineMsg* impl_msg);
 */
    void refineElement(refineMsg* impl_msg) ;

/* DECLS: threaded void refiningElements(void);
 */
    void refiningElements(void) ;

/* DECLS: void coarsenElement(coarsenMsg* impl_msg);
 */
    void coarsenElement(coarsenMsg* impl_msg) ;

/* DECLS: threaded void coarseningElements(void);
 */
    void coarseningElements(void) ;

/* DECLS: sync nodeMsg* getNode(intMsg* impl_msg);
 */

/* DECLS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */

/* DECLS: sync void setBorder(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */

/* DECLS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */

/* DECLS: sync void collapseHelp(collapseMsg* impl_msg);
 */

/* DECLS: void checkPending(refMsg* impl_msg);
 */
    void checkPending(refMsg* impl_msg) ;

/* DECLS: void checkPending(drefMsg* impl_msg);
 */
    void checkPending(drefMsg* impl_msg) ;

/* DECLS: sync void updateElement(updateMsg* impl_msg);
 */

/* DECLS: sync void updateElementEdge(updateMsg* impl_msg);
 */

/* DECLS: void updateReferences(updateMsg* impl_msg);
 */
    void updateReferences(updateMsg* impl_msg) ;

/* DECLS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */

/* DECLS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* setPending(intMsg* impl_msg);
 */

/* DECLS: sync void unsetPending(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* isPending(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* lockNode(intMsg* impl_msg);
 */

/* DECLS: sync void unlockNode(intMsg* impl_msg);
 */

/* DECLS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNotNode(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNotElem(refMsg* impl_msg);
 */

/* DECLS: sync void setTargetArea(doubleMsg* impl_msg);
 */

/* DECLS: sync void resetTargetArea(doubleMsg* impl_msg);
 */

/* DECLS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */

/* DECLS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */

/* DECLS: void reportPos(nodeMsg* impl_msg);
 */
    void reportPos(nodeMsg* impl_msg) ;

/* DECLS: threaded sync void print(void);
 */

/* DECLS: threaded sync void out_print(void);
 */

/* DECLS: void freshen(void);
 */
    void freshen(void) ;

/* DECLS: void deriveBorderNodes(void);
 */
    void deriveBorderNodes(void) ;

/* DECLS: void tweakMesh(void);
 */
    void tweakMesh(void) ;

/* DECLS: void improveChunk(void);
 */
    void improveChunk(void) ;

/* DECLS: threaded sync void improve(void);
 */

/* DECLS: void addNode(nodeMsg* impl_msg);
 */
    void addNode(nodeMsg* impl_msg) ;

/* DECLS: void addEdge(edgeMsg* impl_msg);
 */
    void addEdge(edgeMsg* impl_msg) ;

/* DECLS: void addElement(elementMsg* impl_msg);
 */
    void addElement(elementMsg* impl_msg) ;

/* DECLS: sync void removeNode(intMsg* impl_msg);
 */

/* DECLS: sync void removeEdge(intMsg* impl_msg);
 */

/* DECLS: sync void removeElement(intMsg* impl_msg);
 */

/* DECLS: sync void updateNode(updateMsg* impl_msg);
 */

/* DECLS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
};
PUPmarshall(CProxy_chunk);
/* ---------------- section proxy -------------- */
 class CProxySection_chunk : public CProxySection_ArrayElement{
  public:
    typedef chunk local_t;
    typedef CkIndex_chunk index_t;
    typedef CProxy_chunk proxy_t;
    typedef CProxyElement_chunk element_t;
    typedef CProxySection_chunk section_t;

    CProxySection_chunk(void) {}
    void ckDelegate(CkDelegateMgr *to) {
      CProxySection_ArrayElement::ckDelegate(to);
    }
    void ckUndelegate(void) {
      CProxySection_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxySection_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_SECTION(CProxySection_ArrayElement)
//Generalized array indexing:
    CProxyElement_chunk operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_chunk operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_chunk operator [] (int idx) const 
        {return CProxyElement_chunk(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}
    CProxyElement_chunk operator () (int idx) const 
        {return CProxyElement_chunk(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}
    CProxySection_chunk(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems, CkGroupID dTo) 
        :CProxySection_ArrayElement(aid,elems,nElems,dTo) {}
    CProxySection_chunk(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) 
        :CProxySection_ArrayElement(aid,elems,nElems) {}
    CProxySection_chunk(const CkSectionID &sid)       :CProxySection_ArrayElement(sid) {}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
/* DECLS: chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: chunk(chunkMsg* impl_msg);
 */

/* DECLS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
    void addRemoteEdge(remoteEdgeMsg* impl_msg) ;

/* DECLS: void refineElement(refineMsg* impl_msg);
 */
    void refineElement(refineMsg* impl_msg) ;

/* DECLS: threaded void refiningElements(void);
 */
    void refiningElements(void) ;

/* DECLS: void coarsenElement(coarsenMsg* impl_msg);
 */
    void coarsenElement(coarsenMsg* impl_msg) ;

/* DECLS: threaded void coarseningElements(void);
 */
    void coarseningElements(void) ;

/* DECLS: sync nodeMsg* getNode(intMsg* impl_msg);
 */

/* DECLS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */

/* DECLS: sync void setBorder(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */

/* DECLS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */

/* DECLS: sync void collapseHelp(collapseMsg* impl_msg);
 */

/* DECLS: void checkPending(refMsg* impl_msg);
 */
    void checkPending(refMsg* impl_msg) ;

/* DECLS: void checkPending(drefMsg* impl_msg);
 */
    void checkPending(drefMsg* impl_msg) ;

/* DECLS: sync void updateElement(updateMsg* impl_msg);
 */

/* DECLS: sync void updateElementEdge(updateMsg* impl_msg);
 */

/* DECLS: void updateReferences(updateMsg* impl_msg);
 */
    void updateReferences(updateMsg* impl_msg) ;

/* DECLS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */

/* DECLS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* setPending(intMsg* impl_msg);
 */

/* DECLS: sync void unsetPending(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* isPending(intMsg* impl_msg);
 */

/* DECLS: sync intMsg* lockNode(intMsg* impl_msg);
 */

/* DECLS: sync void unlockNode(intMsg* impl_msg);
 */

/* DECLS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNotNode(refMsg* impl_msg);
 */

/* DECLS: sync refMsg* getNotElem(refMsg* impl_msg);
 */

/* DECLS: sync void setTargetArea(doubleMsg* impl_msg);
 */

/* DECLS: sync void resetTargetArea(doubleMsg* impl_msg);
 */

/* DECLS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */

/* DECLS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */

/* DECLS: void reportPos(nodeMsg* impl_msg);
 */
    void reportPos(nodeMsg* impl_msg) ;

/* DECLS: threaded sync void print(void);
 */

/* DECLS: threaded sync void out_print(void);
 */

/* DECLS: void freshen(void);
 */
    void freshen(void) ;

/* DECLS: void deriveBorderNodes(void);
 */
    void deriveBorderNodes(void) ;

/* DECLS: void tweakMesh(void);
 */
    void tweakMesh(void) ;

/* DECLS: void improveChunk(void);
 */
    void improveChunk(void) ;

/* DECLS: threaded sync void improve(void);
 */

/* DECLS: void addNode(nodeMsg* impl_msg);
 */
    void addNode(nodeMsg* impl_msg) ;

/* DECLS: void addEdge(edgeMsg* impl_msg);
 */
    void addEdge(edgeMsg* impl_msg) ;

/* DECLS: void addElement(elementMsg* impl_msg);
 */
    void addElement(elementMsg* impl_msg) ;

/* DECLS: sync void removeNode(intMsg* impl_msg);
 */

/* DECLS: sync void removeEdge(intMsg* impl_msg);
 */

/* DECLS: sync void removeElement(intMsg* impl_msg);
 */

/* DECLS: sync void updateNode(updateMsg* impl_msg);
 */

/* DECLS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
};
PUPmarshall(CProxySection_chunk);
typedef CBaseT<ArrayElementT<CkIndex1D>,CProxy_chunk>  CBase_chunk;

extern void _registerrefine(void);
#endif
