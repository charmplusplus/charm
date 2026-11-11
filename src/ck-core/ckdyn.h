/*
 * CkDyn: dynamic chares for language bindings.
 *
 * A binding (Python, Julia, a C program, ...) describes its chare classes at
 * run time through tables of entry descriptors, and the runtime reaches the
 * binding's objects through one table of function pointers (CkDynHooks).
 * Every entry method of every dynamic class delivers one message type,
 * CkDynMsg, whose payload is a header plus typed slots (3.4 of the design
 * document). This header is the complete binding-facing API: C types and
 * extern "C" functions only, no Charm++ header is included, so a binding
 * can be built without the Charm++ include tree. The C++ classes behind it
 * are in ckdyn-impl.h.
 *
 * Design and rationale: charm4py-thin DESIGN.md (sections 3.1-3.17).
 * Internals chapter: doc/internals/ckdyn.md.
 */
#ifndef CKDYN_H
#define CKDYN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any incompatible change to this file. */
#define CKDYN_API_VERSION 2

/* Handles. Every handle a binding gives CkDyn (from construct, mainchare or
 * unpack) must be non-NULL: CkDyn uses NULL to mark a branch that was not
 * constructed (a group branch outside a constrained group's PE list). */

/* ------------------------------------------------------------------------ */
/* Classes and entry methods                                                 */
/* ------------------------------------------------------------------------ */

typedef enum {
  CKDYN_MAINCHARE = 1,
  CKDYN_CHARE     = 2,
  CKDYN_GROUP     = 3,
  CKDYN_NODEGROUP = 4,
  CKDYN_ARRAY     = 5,
  CKDYN_ARRAYMAP  = 6
} CkDynKind;

/* Entry-method attributes. The names match the .ci attributes. */
enum {
  CKDYN_EP_THREADED   = 1u << 0,  /* the binding runs it on a coroutine; passed through in ep_flags */
  CKDYN_EP_INLINE     = 1u << 1,  /* CK_EP_INLINE: a local send runs the method at once */
  CKDYN_EP_EXPEDITED  = 1u << 2,  /* every send to it gets CK_MSG_EXPEDITED */
  CKDYN_EP_IMMEDIATE  = 1u << 3,  /* CK_EP_IMMEDIATE (nodegroups only) */
  CKDYN_EP_NOKEEP     = 1u << 4,  /* always set by CkDyn; see the ownership rules below */
  CKDYN_EP_NOTRACE    = 1u << 5,  /* CK_EP_TRACEDISABLE */
  CKDYN_EP_APPWORK    = 1u << 6,  /* CK_EP_APPWORK */
  CKDYN_EP_EXCLUSIVE  = 1u << 7,  /* nodegroups: one invocation at a time per branch */
  CKDYN_EP_REDNTARGET = 1u << 8   /* documentation only: may be a reduction target */
};

typedef struct CkDynEntryDesc {
  const char *name;     /* "Class.method"; copied at registration */
  uint32_t    flags;    /* CKDYN_EP_* */
  uint32_t    user_id;  /* opaque to CkDyn; passed back on every delivery */
} CkDynEntryDesc;

struct CkDynHooks;

typedef struct CkDynClassDesc {
  const char            *name;         /* "Class"; copied at registration */
  CkDynKind              kind;
  int                    ndims;        /* arrays only: 1..6, or 0 for any (the index given at creation decides) */
  int                    num_entries;  /* entries[0] is the constructor */
  const CkDynEntryDesc  *entries;
  void                  *binding_data; /* opaque; passed to every per-class hook */
  const struct CkDynHooks *hooks;      /* NULL: the table given to CkDynInit */
} CkDynClassDesc;

/* ------------------------------------------------------------------------ */
/* Targets: one element of any kind                                          */
/* ------------------------------------------------------------------------ */

#define CKDYN_MAX_DIMS 6

/* Structs that CkDyn copies as bytes into messages (CkDynTarget,
 * CkDynSectionID, CkDynCallback, CkDynHdr, CkDynSlot) use fixed-width integer
 * fields only; the enums supply the constants. ckdyn.C checks their sizes. */
typedef struct CkDynTarget {
  int32_t kind;                        /* CkDynKind */
  int32_t chare_idx;                   /* the Charm++ chare index of its class */
  union {
    struct { int32_t pe; void *ptr; } chare;            /* MAINCHARE, CHARE: CkChareID */
    struct { int32_t gid; int32_t pe_or_node; } group;  /* GROUP, NODEGROUP, ARRAYMAP */
    struct { int32_t aid; int32_t ndims; int32_t index[CKDYN_MAX_DIMS]; } array;   /* ARRAY */
  } u;
} CkDynTarget;                         /* 40 bytes */

/* ------------------------------------------------------------------------ */
/* Section ids                                                               */
/* ------------------------------------------------------------------------ */

typedef struct CkDynSectionID {
  int32_t  root_pe;    /* PE whose CkDynMgr holds the section (the PE that created it) */
  int32_t  idx;        /* index in that table */
  int32_t  coll_id;    /* array or group id (-1 for a combined section) */
  uint8_t  kind;       /* CKDYN_ARRAY, CKDYN_GROUP, or 0 for a combined section (CkDynSectionCombine) */
  uint8_t  reserved;
  uint16_t gen;        /* generation, so a reused slot is not mistaken for the old section */
} CkDynSectionID;                     /* 16 bytes */

/* ------------------------------------------------------------------------ */
/* Callbacks                                                                 */
/* ------------------------------------------------------------------------ */

typedef enum {
  CKDYN_CB_TARGET  = 1,   /* an entry method of one element, or a broadcast to a collection */
  CKDYN_CB_SECTION = 2,   /* an entry method multicast to a section (relayed by CkDynMgr; a pup-able callback) */
  CKDYN_CB_FUTURE  = 3,   /* a binding future (pe, id) */
  CKDYN_CB_IGNORE  = 4,
  CKDYN_CB_EXIT    = 5    /* CkExit */
} CkDynCbKind;

/* A TARGET callback to a main chare entry is routed by chare index through
 * CkDynMgr on the main chare's PE, not by the target's object pointer, so it
 * stays valid in a program restarted from a checkpoint (CkDynCheckpoint). */
typedef struct CkDynCallback {
  int32_t        kind;     /* CkDynCbKind */
  CkDynTarget    t;        /* TARGET: the element, or the collection when bcast != 0 */
  int32_t        ep;       /* TARGET, SECTION */
  int32_t        bcast;
  uint32_t       refnum;   /* non-zero: the binding future the delivered result belongs to */
  CkDynSectionID sid;      /* SECTION */
  int32_t        pe;       /* FUTURE: the PE that owns the future */
  uint32_t       id;       /* FUTURE */
} CkDynCallback;           /* 88 bytes */

/* ------------------------------------------------------------------------ */
/* Messages                                                                  */
/* ------------------------------------------------------------------------ */

/* Element kinds of typed buffers and of reduction data. */
enum {
  CKDYN_ELEM_NONE = 0,
  CKDYN_ELEM_I8, CKDYN_ELEM_I16, CKDYN_ELEM_I32, CKDYN_ELEM_I64,
  CKDYN_ELEM_U8, CKDYN_ELEM_U16, CKDYN_ELEM_U32, CKDYN_ELEM_U64,
  CKDYN_ELEM_F16, CKDYN_ELEM_F32, CKDYN_ELEM_F64,
  CKDYN_ELEM_C64, CKDYN_ELEM_C128,
  CKDYN_ELEM_BOOL, CKDYN_ELEM_BYTE
};

/* Slot types. */
enum {
  CKDYN_SLOT_NONE = 0,  /* the binding's null */
  CKDYN_SLOT_BOOL,
  CKDYN_SLOT_I64,
  CKDYN_SLOT_U64,
  CKDYN_SLOT_F64,
  CKDYN_SLOT_BYTES,     /* small blob inline in the data region */
  CKDYN_SLOT_BUFFER,    /* typed n-dimensional buffer copied into the data region */
  CKDYN_SLOT_NCPY,      /* typed host buffer transferred by the runtime, not copied into the message */
  CKDYN_SLOT_DEVICE,    /* typed device buffer, transferred device to device */
  CKDYN_SLOT_FUTURE,    /* (pe, id) of a binding future */
  CKDYN_SLOT_CALLBACK,  /* a CkDynCallback in the data region */
  CKDYN_SLOT_SECTION,   /* a CkDynSectionID in the data region */
  CKDYN_SLOT_PROXY,     /* a CkDynTarget in the data region */
  CKDYN_SLOT_TAIL       /* marks the opaque tail; len = tail_len */
};

/* Buffer flags (CkDynSlot.bflags, CkDynBufDesc.bflags). */
enum {
  CKDYN_BUF_FORTRAN  = 1u << 0,  /* column-major */
  CKDYN_BUF_READONLY = 1u << 1,
  CKDYN_BUF_OOB      = 1u << 2   /* came from the binding's serializer out of band */
};

/* Header flags (CkDynHdr.flags). */
enum {
  CKDYN_MSG_BLOCK      = 1u << 0,  /* a FUTURE slot receives the method's return value */
  CKDYN_MSG_BCAST      = 1u << 1,
  CKDYN_MSG_BCASTRET   = 1u << 2,  /* gather the return values of a broadcast */
  CKDYN_MSG_CREATION   = 1u << 3,  /* constructor message that acknowledges creation */
  CKDYN_MSG_USESATSYNC = 1u << 4,  /* array element created with usesAtSync */
  CKDYN_MSG_RAY        = 1u << 5,
  CKDYN_MSG_CONSTRAINED = 1u << 6, /* group constructor: the last slot is a BYTES slot of int32 PE numbers;
                                      branches on other PEs are not constructed (handle stays NULL) */
  CKDYN_MSG_BINDING_MASK = 0xF000u /* bits 12-15: the binding's own flags; CkDyn neither sets nor reads them */
};

#define CKDYN_MAGIC 0xDA01u

typedef struct CkDynHdr {
  uint16_t magic_version;   /* CKDYN_MAGIC */
  uint16_t ep;              /* Charm++ entry index; authoritative (the envelope's copy can be stale on inline sends) */
  uint16_t flags;           /* CKDYN_MSG_* */
  uint16_t nslots;
  uint32_t user_id;         /* the entry's user_id, or the binding reducer id for foreign-reducer contributions */
  uint32_t tail_len;        /* bytes of opaque tail */
  uint32_t data_len;        /* bytes of the inline data region */
  uint32_t nbuf;            /* number of NCPY/DEVICE slots */
} CkDynHdr;                 /* 24 bytes */

typedef struct CkDynSlot {  /* 24 bytes, nslots of them right after the header */
  uint8_t  type;            /* CKDYN_SLOT_* */
  uint8_t  elem;            /* CKDYN_ELEM_* for BUFFER/NCPY/DEVICE */
  uint8_t  ndim;            /* 0..CKDYN_MAX_BUF_DIMS for buffers, 0 for scalars */
  uint8_t  bflags;          /* CKDYN_BUF_* */
  uint16_t tag;             /* binding-defined subtype, opaque to CkDyn */
  uint16_t reserved;        /* DEVICE: the device ordinal of the data (the receiver's device after delivery) */
  uint32_t len;             /* bytes (BYTES/BUFFER/NCPY/DEVICE/TAIL), else 0 */
  uint32_t reserved2;       /* NCPY/DEVICE: the slot's descriptor index (CkDyn's own) */
  union {
    int64_t  i64;
    uint64_t u64;
    double   f64;
    uint64_t off;           /* offset into the data region (BYTES, BUFFER, NCPY, DEVICE, CALLBACK, SECTION, PROXY);
                               for BUFFER the shape (ndim int64) precedes the bytes at that offset;
                               for NCPY and DEVICE only the shape is there (read the data with CkDynViewBufferPtr) */
  } v;
} CkDynSlot;

#define CKDYN_MAX_BUF_DIMS 32

/* Payload layout, from CkDynMsg::buf:
 *   CkDynHdr, CkDynSlot[nslots], data region (8-byte aligned), tail (tail_len bytes).
 * The data region holds BYTES payloads, BUFFER shapes and contents, and the
 * pupped bytes of CALLBACK, SECTION and PROXY values.
 */

/* A buffer described to the builder. */
typedef struct CkDynBufDesc {
  const void    *ptr;
  uint64_t       len;        /* bytes */
  uint8_t        elem;       /* CKDYN_ELEM_* */
  uint8_t        ndim;       /* 0..CKDYN_MAX_BUF_DIMS */
  uint8_t        bflags;     /* CKDYN_BUF_* */
  uint16_t       tag;
  const int64_t *shape;      /* ndim entries; NULL when ndim == 0 */
  const int64_t *strides;    /* ndim entries in bytes, or NULL for C-contiguous; must describe a contiguous layout */
} CkDynBufDesc;

typedef struct CkDynMsgBuilder CkDynMsgBuilder;      /* opaque */
typedef struct CkDynMsg CkDynMsg;                    /* opaque to C: the Charm++ message */
typedef struct CkDynMcastMsg CkDynMcastMsg;          /* opaque to C: the CkMulticast message (CkDynSendSection) */
typedef void (*CkDynBufDone)(void *done_arg);        /* the source buffer may be released (NCPY, DEVICE) */

/* Begin a message. Hints size the scratch space; prio_bits reserves that many
 * priority bits (0: none); group_dep is a CkGroupID index the message must
 * wait for on delivery (-1: none). Group dependencies must be known at
 * allocation (the envelope reserves space for them), which is why they are
 * here and not in CkDynSendOpts. The runtime honors them on chare, group,
 * nodegroup and creation messages, not on point-to-point array element sends. */
CkDynMsgBuilder *CkDynMsgBegin(int nslots_hint, size_t data_hint, size_t tail_hint, int prio_bits, int group_dep);
void  CkDynMsgSetFlags(CkDynMsgBuilder *b, uint16_t flags);
void  CkDynMsgSetUserId(CkDynMsgBuilder *b, uint32_t user_id);
void  CkDynMsgAddNone(CkDynMsgBuilder *b);
void  CkDynMsgAddBool(CkDynMsgBuilder *b, int v);
void  CkDynMsgAddI64(CkDynMsgBuilder *b, int64_t v);
void  CkDynMsgAddU64(CkDynMsgBuilder *b, uint64_t v);
void  CkDynMsgAddF64(CkDynMsgBuilder *b, double v);
void  CkDynMsgAddBytes(CkDynMsgBuilder *b, const void *p, uint32_t len, uint16_t tag);
/* The shape is copied now; the bytes are copied into the message by
 * CkDynMsgFinish, so d->ptr must stay valid until Finish or Abandon. In the
 * reduce hook's result builder the bytes are copied now as well, so d->ptr
 * may be freed as soon as this returns. */
void  CkDynMsgAddBuffer(CkDynMsgBuilder *b, const CkDynBufDesc *d);
void  CkDynMsgAddFuture(CkDynMsgBuilder *b, int pe, uint32_t id);
void  CkDynMsgAddCallback(CkDynMsgBuilder *b, const CkDynCallback *cb);
void  CkDynMsgAddSection(CkDynMsgBuilder *b, const CkDynSectionID *sid);
void  CkDynMsgAddProxy(CkDynMsgBuilder *b, const CkDynTarget *t);
void *CkDynMsgReserveTail(CkDynMsgBuilder *b, uint32_t len);   /* the binding writes len bytes in place */
/* Priority bits: valid after CkDynMsgFinish; see CkDynMsgPriorityPtr. */
CkDynMsg *CkDynMsgFinish(CkDynMsgBuilder *b, int ep);          /* one allocation; ep goes in the header */
/* The same payload as a section message, for CkDynSendSection. It takes no
 * group dependency; CkDynMsgPriorityPtr, CkDynMsgSetQueueing and CkDynMsgFree
 * accept it cast to CkDynMsg *. */
CkDynMcastMsg *CkDynMsgFinishMcast(CkDynMsgBuilder *b, int ep);
void  CkDynMsgAbandon(CkDynMsgBuilder *b);
void *CkDynMsgPriorityPtr(CkDynMsg *m);                        /* the reserved priority bits, to fill in */
/* Queueing strategies for CkDynMsgSetQueueing; the values equal Charm++'s
 * CK_QUEUEING_* (checked in ckdyn.C). The default without a call is FIFO. */
enum {
  CKDYN_QUEUEING_FIFO  = 2,
  CKDYN_QUEUEING_LIFO  = 3,
  CKDYN_QUEUEING_IFIFO = 4,  /* integer priority, lower first */
  CKDYN_QUEUEING_ILIFO = 5,
  CKDYN_QUEUEING_BFIFO = 6,  /* bitvector priority */
  CKDYN_QUEUEING_BLIFO = 7,
  CKDYN_QUEUEING_LFIFO = 8,  /* long integer priority */
  CKDYN_QUEUEING_LLIFO = 9
};
void  CkDynMsgSetQueueing(CkDynMsg *m, int queueing);          /* a CKDYN_QUEUEING_* value */
void  CkDynMsgFree(CkDynMsg *m);                               /* a message that will not be sent */
/* The payload bytes of a finished message (header, slots, data, tail), for a
 * foreign-reducer contribution (CkDynContribute with CkDynForeignReducerType). */
const void *CkDynMsgPayload(CkDynMsg *m, uint32_t *len);

/* Reading a delivered message. */
typedef enum {
  CKDYN_MSGKIND_DYN = 0,     /* a CkDynMsg */
  CKDYN_MSGKIND_MCAST,       /* a CkDynMcastMsg: a section send or a section callback's result */
  CKDYN_MSGKIND_REDRESULT,   /* a CkReductionMsg, viewed with one synthesized BUFFER slot */
  CKDYN_MSGKIND_SYS          /* a system message: no slots, except FUTURE when a refnum is set */
} CkDynMsgKind;
/* The slots of a REDRESULT or SYS view are synthesized in the caller's stack
 * frame: the view, its slots and the pointers read from it are valid only
 * until the hook that received the view returns. CkDynMsgRetain keeps the
 * message (the reduction data) alive, not the synthesized slots. */

typedef struct CkDynMsgView {
  CkDynMsgKind          kind;
  const CkDynHdr       *hdr;        /* NULL for REDRESULT and SYS */
  const CkDynSlot      *slots;
  int                   nslots;
  const char           *data;       /* data region */
  const char           *tail;       /* tail_len bytes, or NULL */
  uint32_t              tail_len;
  uint32_t              refnum;     /* the envelope's reference number (0 if none; the all-ones value, the
                                       default reduction user flag, also reads as none); for a REDRESULT or
                                       SYS view a FUTURE slot (CkMyPe(), refnum) is synthesized too; a foreign
                                       reduction result is viewed as DYN and carries it only here */
  const CkDynSectionID *section;    /* non-NULL for an MCAST delivery */
  void                 *msg;        /* the runtime message, for retain */
} CkDynMsgView;

/* Pointer to a BUFFER, NCPY or DEVICE slot's bytes; *shape_out gets ndim int64
 * values. For a DEVICE slot the pointer is device memory on the device the
 * slot's `reserved` field names. NCPY and DEVICE data live as long as the
 * message (CkDynMsgRetain), unless the slot was delivered into a posted buffer. */
const void *CkDynViewBufferPtr(const CkDynMsgView *v, int slot, const int64_t **shape_out);
const void *CkDynViewBytesPtr(const CkDynMsgView *v, int slot);
void        CkDynViewCallback(const CkDynMsgView *v, int slot, CkDynCallback *out);
void        CkDynViewSection(const CkDynMsgView *v, int slot, CkDynSectionID *out);
void        CkDynViewProxy(const CkDynMsgView *v, int slot, CkDynTarget *out);

/* Ownership. Every CkDyn entry is registered CK_EP_NOKEEP, so the runtime frees
 * the message after the hook returns. A binding that keeps pointers into the
 * message (a zero-copy view of a BUFFER slot) calls CkDynMsgRetain before
 * returning and CkDynMsgRelease once per retain when done. */
void CkDynMsgRetain(const CkDynMsgView *v);
void CkDynMsgRelease(void *msg);

/* ------------------------------------------------------------------------ */
/* The hook table                                                            */
/* ------------------------------------------------------------------------ */

enum {
  CKDYN_PUP_MIGRATE    = 1,
  CKDYN_PUP_CHECKPOINT = 2,
  CKDYN_PUP_SIZING     = 0x100   /* OR'd into reason on the sizing pass */
};
#define CKDYN_PACK_REFUSED ((size_t)-1)   /* pack_size: the object cannot be packed in this state; CkDyn aborts */

typedef struct CkDynHooks {
  uint32_t version;              /* CKDYN_API_VERSION */
  uint32_t size;                 /* sizeof(CkDynHooks) as compiled by the binding */
  void *global;                  /* passed to the hooks that have no class: register_classes, readonly, reduce, future, error */
  void (*register_classes)(void *global);   /* called at start-up in place of CkRegisterMainModule (shared library) */
  /* object lifecycle */
  void *(*construct)(void *binding_data, const CkDynTarget *t, const CkDynMsgView *ctor_msg);
  void  (*destruct)(void *binding_data, void *handle);
  /* delivery */
  void  (*deliver)(void *binding_data, void *handle, int ep, uint32_t user_id,
                   uint32_t ep_flags, const CkDynMsgView *msg);
  /* array broadcast: one call per PE for every local element whose epoch matches.
     Time is charged to the array manager, and tracing records no event per element,
     unless the binding brackets each element's call with CkDynCoroResume/Suspend
     (or CkDynCallstackPush/Pop for the load alone). A handler that destroys another local element
     makes that element's handle invalid for the rest of the call. */
  void  (*deliver_many)(void *binding_data, void **handles, int n, int ep, uint32_t user_id,
                        uint32_t ep_flags, const CkDynMsgView *msg);
  void  (*mainchare)(void *binding_data, void **handle_out, const CkDynTarget *t, int argc, char **argv);
  /* pup */
  size_t (*pack_size)(void *binding_data, void *handle, int reason);
  void   (*pack)(void *binding_data, void *handle, int reason, void *buf, size_t len);
  void  *(*unpack)(void *binding_data, const CkDynTarget *t, int reason, const void *buf, size_t len);
  size_t (*pack_readonly_size)(void *global);
  void   (*pack_readonly)(void *global, void *buf, size_t len);
  void   (*unpack_readonly)(void *global, const void *buf, size_t len);
  /* reductions: runs inside the reduction manager, no class context */
  void  (*reduce)(void *global, uint32_t reducer_id, const CkDynMsgView **contribs, int n,
                  CkDynMsgBuilder *result);
  /* futures */
  void  (*future)(void *global, int pe, uint32_t id, const CkDynMsgView *value);
  /* sections: a member received its section's creation multicast (optional) */
  void  (*section_joined)(void *binding_data, void *handle, const CkDynSectionID *sid);
  /* array map */
  int   (*map_proc_num)(void *binding_data, void *handle, int aid, int ndims, const int *index);
  /* load balancing */
  void  (*resume_from_sync)(void *binding_data, void *handle);
  /* errors raised inside CkDyn on behalf of the binding; CkDyn aborts after it returns */
  void  (*error)(void *global, const char *what);
} CkDynHooks;

/* ------------------------------------------------------------------------ */
/* Start-up and registration                                                 */
/* ------------------------------------------------------------------------ */

int  CkDynApiVersion(void);
int  CkDynRefnumBytes(void);          /* sizeof(CMK_REFNUM_TYPE) of this libcharm */
int  CkDynHasDeviceSupport(void);
/* The device HAPI assigned to the calling PE (-1 without CUDA or HIP), and
 * the same made the calling thread's current device (cudaSetDevice), so
 * that a binding's libraries allocate on it by default; returns the device. */
int  CkDynHapiMyDevice(void);
int  CkDynHapiUseMyDevice(void);
/* Install the hook table (copied). version must equal CKDYN_API_VERSION and
 * size sizeof(CkDynHooks), otherwise CkDyn aborts; the same check applies to
 * a class's own table (CkDynClassDesc.hooks). A shared-library binding calls it before
 * CkDynStart; a C++ program calls it from an initnode function, before it
 * registers classes. */
void CkDynInit(const CkDynHooks *hooks);
/* Start the runtime (ConverseInit); never returns. Shared-library bindings only. */
void CkDynStart(int argc, char **argv);
/* Create the CkDynMgr group. MainchareDyn does this itself; a C++ main chare
 * that uses CkDyn without MainchareDyn calls it from its constructor. */
void CkDynCreateManagers(void);
/* Register a class; returns the chare index. *first_ep receives the entry index
 * of entries[0]; entries[i] has index first_ep + i. Call on rank 0 of every
 * process, in the same order, before start-up completes (from
 * hooks.register_classes or an initnode). */
int  CkDynRegisterClass(const CkDynClassDesc *desc, int *first_ep);
void CkDynAllowLateRegistration(void);
int  CkDynRegisterClassLate(const CkDynClassDesc *desc, int *first_ep);
/* The chare index of a dynamic target's class, and its descriptor. */
const CkDynClassDesc *CkDynClassOf(int chare_idx);

/* ------------------------------------------------------------------------ */
/* Sends                                                                     */
/* ------------------------------------------------------------------------ */

enum {
  CKDYN_SEND_INLINE    = 1u << 0,
  CKDYN_SEND_EXPEDITED = 1u << 1,
  CKDYN_SEND_IMMEDIATE = 1u << 2,
  CKDYN_SEND_NOTRACE   = 1u << 3
};

typedef struct CkDynSendOpts {
  uint32_t flags;     /* CKDYN_SEND_*; the entry's own attributes are added by CkDyn */
} CkDynSendOpts;      /* NULL means no flags */

void CkDynSendChare     (const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o);
void CkDynSendGroup     (int gid, int pe, int ep, CkDynMsg *m, const CkDynSendOpts *o);          /* pe == -1: broadcast */
void CkDynSendGroupMulti(int gid, int npes, const int *pes, int ep, CkDynMsg *m, const CkDynSendOpts *o);
void CkDynSendNodeGroup (int gid, int node, int ep, CkDynMsg *m, const CkDynSendOpts *o);        /* node == -1: broadcast */
void CkDynSendArray     (int aid, int ndims, const int *index, int ep, CkDynMsg *m, const CkDynSendOpts *o); /* index == NULL: broadcast */
void CkDynSendSection   (const CkDynSectionID *sid, int ep, CkDynMcastMsg *m, const CkDynSendOpts *o);     /* opts ignored */
/* Send to whatever a target names (chare, group branch, nodegroup branch, array element). */
void CkDynSendTarget    (const CkDynTarget *t, int ep, CkDynMsg *m, const CkDynSendOpts *o);

/* ------------------------------------------------------------------------ */
/* Creation                                                                  */
/* ------------------------------------------------------------------------ */

typedef struct CkDynArrayOpts {
  int        ndims;
  const int *dims;               /* NULL: empty array, insert later */
  int        map_gid;            /* -1: default map */
  int        anytime_migration;  /* CkArrayOptions::setAnytimeMigration */
  int        static_insertion;   /* CkArrayOptions::setStaticInsertion */
  int        bind_to_aid;        /* -1, or an array to bindTo */
} CkDynArrayOpts;
/* A group the creation must wait for (for example the array's map, see the
 * bindTo race in charm_best_practices.md) is the group_dep of the constructor
 * message, given to CkDynMsgBegin. */

#define CKDYN_PE_ANY (-1)

/* *out receives a virtual chare id (pe < 0) usable for sends and callbacks at once;
 * hooks.construct receives the real id. Both name the same chare. */
void CkDynCreateChare       (int chare_idx, int on_pe, CkDynMsg *ctor, CkDynTarget *out);   /* on_pe CKDYN_PE_ANY: seed balancer */
/* A constrained group (branches on a PE list only) is a group whose constructor
 * message carries CKDYN_MSG_CONSTRAINED and the PE list as its last slot. */
int  CkDynCreateGroup       (int chare_idx, CkDynMsg *ctor);
int  CkDynCreateNodeGroup   (int chare_idx, CkDynMsg *ctor);
int  CkDynCreateArray       (int chare_idx, CkDynMsg *ctor, const CkDynArrayOpts *o);        /* returns aid */
/* The constructor message must have been finished with the class's constructor
 * entry index (first_ep), which names the class to construct; every element of
 * one array must be of the array's class. CKDYN_MSG_CREATION is for whole-array
 * creation only: on an insert it would start a one-element reduction. */
void CkDynArrayInsert       (int aid, int ndims, const int *index, int on_pe, CkDynMsg *ctor);
void CkDynArrayDoneInserting(int aid);
void CkDynArrayDestroyElement(int aid, int ndims, const int *index);                         /* local element only */

/* ------------------------------------------------------------------------ */
/* Futures and callbacks                                                     */
/* ------------------------------------------------------------------------ */

/* Deliver a value to a binding future: a CkDynMsg sent to CkDynMgr[pe], which
 * calls hooks.future(global, pe, id, view). Future ids are 1 .. 2^(8*CkDynRefnumBytes())-2:
 * 0 means no reference number and the all-ones value is the reduction user flag. */
void CkDynFutureSet(int pe, uint32_t id, CkDynMsg *value);
/* Send a message through a callback description. */
void CkDynCallbackSend(const CkDynCallback *cb, CkDynMsg *m);
/* The reduction number of the local array element, group or nodegroup branch
 * the target names (an array map has none). */
int  CkDynGetReductionNumber(const CkDynTarget *t);

/* ------------------------------------------------------------------------ */
/* Reductions                                                                */
/* ------------------------------------------------------------------------ */

typedef struct CkDynReducerInfo {
  const char *name;   /* "sum_int", ... as in CkReduction::reducerType */
  int         type;   /* the CkReduction::reducerType value */
  uint8_t     elem;   /* CKDYN_ELEM_* of the data, or CKDYN_ELEM_BYTE for set/concat/statistics/tuple */
} CkDynReducerInfo;

int  CkDynReducers(const CkDynReducerInfo **out);   /* count; *out points at a static table */
int  CkDynForeignReducerType(void);                  /* the one reducerType for binding-defined reducers */

typedef struct CkDynRedData {
  const void *data;
  uint32_t    len;    /* bytes; 0 with data NULL for an empty contribution */
  int         type;   /* CkReduction::reducerType, or CkDynForeignReducerType() */
} CkDynRedData;

/* contribute() on the local element or branch the target names. For a foreign
 * reducer the data is a CkDyn payload (header + slots + tail) whose user_id is
 * the binding's reducer id; hooks.reduce combines them. */
void CkDynContribute(const CkDynTarget *me, const CkDynRedData *d, const CkDynCallback *cb);

/* ------------------------------------------------------------------------ */
/* Sections (doc/internals/ckdyn.md, "Sections")                            */
/* ------------------------------------------------------------------------ */

/* Creation runs on the calling PE, which becomes the root. done receives an
 * empty reduction once every member has its cookie; send after it fires.
 * Combine: the union of array or combined sections (each element once),
 * all with one index dimension count; group parts abort. */

void CkDynSectionCreateArray(int aid, int nelems, int ndims, const int *indices, int bfactor,
                             const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCreateGroup(int gid, int npes, const int *pes, int bfactor,
                             const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionCombine(int nparts, const CkDynSectionID *parts, const CkDynCallback *done, CkDynSectionID *out);
void CkDynSectionDelete(const CkDynSectionID *sid);
int  CkDynSectionRedNo(const CkDynTarget *me, const CkDynSectionID *sid);
void CkDynContributeSection(const CkDynTarget *me, const CkDynSectionID *sid, const CkDynRedData *d,
                            const CkDynCallback *cb);

/* ------------------------------------------------------------------------ */
/* Checkpoint, quiescence, load balancing, tracing                           */
/* ------------------------------------------------------------------------ */

/* CkStartCheckpoint. done is called when the checkpoint is written and, in a
 * program restarted with +restart, once restart completes. It may be a TARGET
 * callback to a group, array element or main chare entry; a FUTURE callback
 * does not survive a restart (the future belongs to the checkpointed run). */
void CkDynCheckpoint(const char *dirname, const CkDynCallback *done);
void CkDynStartQD(const CkDynCallback *cb);

void CkDynAtSync(const CkDynTarget *t);
void CkDynSetMigratable(const CkDynTarget *t, int migratable);
void CkDynMigrateMe(const CkDynTarget *t, int to_pe);
void CkDynLBInstrument(int on);
void CkDynSetObjLoad(const CkDynTarget *t, double secs);
/* Push/Pop the running object; always paired: Pop pops one entry even when the
   object destroyed itself after Push. */
void CkDynCallstackPush(const CkDynTarget *t);
void CkDynCallstackPop(const CkDynTarget *t);
/* Bracket one run of a binding coroutine (a resumed greenlet) so that its
 * time is charged to the object it belongs to. Call Resume right after
 * switching into the coroutine and Suspend right before switching out (or
 * when it finishes). Resume pushes the target's object on the running-object
 * stack (load balancer timing) and, when tracing is on, begins an execute
 * event for entry ep; Suspend ends both. The trace event's object id is the
 * element's index for an array element and is made from coro_id otherwise.
 * Call them outside any traced entry: from a hook that runs inside
 * CkDynMgr.deposit (not traced) or another untraced entry, so that the
 * events do not nest. They also serve to trace each element of a
 * deliver_many call. */
void CkDynCoroResume(const CkDynTarget *t, int ep, uint64_t coro_id);
void CkDynCoroSuspend(const CkDynTarget *t);

int  CkDynTraceRegisterUserEvent(const char *desc, int id);
void CkDynTraceBeginUserBracketEvent(int id);
void CkDynTraceEndUserBracketEvent(int id);
void CkDynTraceBeginEntry(int ep);
void CkDynTraceEndEntry(int ep);

/* ------------------------------------------------------------------------ */
/* Zero-copy and devices                                                     */
/* ------------------------------------------------------------------------ */

/* Mechanism: doc/internals/ckdyn.md, "Zero-copy". */

typedef struct CkDynDevDesc {
  const void *ptr;           /* device memory, C-contiguous */
  uint64_t    len;           /* bytes */
  uint8_t     elem;
  uint8_t     ndim;
  uint16_t    tag;
  int         device;        /* device ordinal */
  uint64_t    stream;        /* the cudaStream_t / hipStream_t the data is valid on (0: the default stream) */
  const int64_t *shape;
} CkDynDevDesc;

/* A slot whose data the runtime transfers without copying it into the
 * message. d->ptr must stay valid until done(arg) runs; it runs exactly once,
 * from the scheduler of the sending PE, after the data has left the source,
 * or after CkDynMsgAbandon / CkDynMsgFree of an unsent message. A message
 * with NCPY or DEVICE slots (header nbuf > 0) is sent point to point only:
 * a broadcast, section, multi-branch, constructor or reduction use aborts.
 * NCPY and DEVICE slots cannot share a message. CkDynMsgAddDevice aborts in
 * a build without CUDA or HIP. */
void CkDynMsgAddNcpy(CkDynMsgBuilder *b, const CkDynBufDesc *d, CkDynBufDone done, void *arg);
void CkDynMsgAddDevice(CkDynMsgBuilder *b, const CkDynDevDesc *d, CkDynBufDone done, void *arg);
/* A destination, on the calling PE, for slot `slot` of the next message to
 * the local object t's entry ep. The lengths must match; one post per
 * (object, ep, slot) at a time; posts are dropped when the object migrates or
 * is destroyed. Not for nodegroups. */
void CkDynPostBuffer(const CkDynTarget *t, int ep, int slot, const CkDynBufDesc *d);
void CkDynPostDeviceBuffer(const CkDynTarget *t, int ep, int slot, const CkDynDevDesc *d);
/* `slot` of a post or unpost OR'd with CKDYN_POST_ARG counts arguments
 * rather than slots: in a message whose header has CKDYN_MSG_BLOCK and whose
 * slot 0 is a FUTURE slot, argument k is slot k + 1, otherwise slot k. The
 * binding then need not know at post time whether the sender will block. */
#define CKDYN_POST_ARG 0x40000000
/* Remove the post for (t, ep, slot) if it is still outstanding (a message
 * whose slot was a BUFFER or pickled value does not consume it). Returns 1
 * when a post was removed, 0 when there was none. */
int  CkDynUnpostBuffer(const CkDynTarget *t, int ep, int slot);

/* Persistent device buffers (doc/internals/ckdyn.md, "Persistent device
 * buffers"). A buffer is opened on one PE and stays registered until closed;
 * its handle, a fixed number of bytes, may be sent to any PE inside a
 * message, and a get or put between an open buffer and a handle needs no
 * entry method and no per-transfer registration. Transfer mode: a copy on
 * the device when both buffers are in one process, CUDA/HIP IPC between
 * processes on one host when the peer's IPC handle opens in this process
 * (+gpushm is not needed), RDMA on registered device memory otherwise. `done` receives an empty message on
 * the issuing PE once per completed get or put: for a FUTURE callback the
 * call's refnum, when non-zero, is the future id; for any other kind it is
 * the callback's refnum (the envelope's reference number). A get is complete
 * when the bytes are in dst; a put when they are in the remote buffer (between
 * hosts the put is carried out as a get issued by the destination's PE).
 * The caller keeps the memory valid and does not touch the destination until
 * done; d->stream orders the transfer after earlier work on it (0: the
 * default stream). Every function aborts in a build without CUDA or HIP. */
typedef struct CkDynDevPersist CkDynDevPersist;      /* opaque, PE-local */
CkDynDevPersist *CkDynDevPersistOpen(const CkDynDevDesc *d, const CkDynCallback *done);
void   CkDynDevPersistClose(CkDynDevPersist *p);      /* aborts with a transfer outstanding */
size_t CkDynDevPersistHandleSize(void);
void   CkDynDevPersistHandle(const CkDynDevPersist *p, void *out);   /* HandleSize() bytes */
/* dst must be open on the calling PE; src_handle names a buffer opened anywhere.
 * dst must hold at least the source's bytes (get) and the destination at least
 * src's (put); the transfer moves the source's length. */
void   CkDynDevPersistGet(CkDynDevPersist *dst, const void *src_handle, uint32_t refnum);
void   CkDynDevPersistPut(const void *dst_handle, CkDynDevPersist *src, uint32_t refnum);
/* The transfer mode a get or put between p and the handle would use:
 * 0 copy in one process, 1 IPC, 2 RDMA. */
int    CkDynDevPersistMode(const CkDynDevPersist *p, const void *handle);
/* cb receives an empty message, from this PE's scheduler, when the stream
 * reaches this point. Aborts in a build without CUDA or HIP. */
void CkDynHapiAddCallback(uint64_t stream, const CkDynCallback *cb);

/* ------------------------------------------------------------------------ */
/* Runtime wrappers, so that a binding includes no Charm++ header            */
/* ------------------------------------------------------------------------ */

int    CkDynMyPe(void);
int    CkDynNumPes(void);
int    CkDynMyNode(void);
int    CkDynNumNodes(void);
int    CkDynMyRank(void);
int    CkDynNodeSize(int node);
int    CkDynNodeFirst(int node);
int    CkDynNodeOf(int pe);
void   CkDynExit(int code);
void   CkDynAbort(const char *msg);
void   CkDynPrintf(const char *text);
double CkDynWallTimer(void);
int    CkDynNumPhysicalNodes(void);
int    CkDynPhysicalNodeID(int pe);
int    CkDynPhysicalRank(int pe);
int    CkDynNumPesOnPhysicalNode(int node);
int    CkDynGetPesOnPhysicalNode(int node, int *pes, int max);
int    CkDynTopoTreeEdges(int pe, int root, const int *pes, int npes, int bfactor,
                          int *parent, int *children, int max);
void   CkDynCallAfter(void (*fn)(void *arg, double now), void *arg, double msecs);
int    CkDynIsRestarting(void);
int    CkDynHasLB(void);

/* CCS */
typedef void (*CkDynCcsHandler)(void *global, const char *name, const void *data, int len);
void   CkDynCcsRegister(const char *name, CkDynCcsHandler fn);
int    CkDynCcsIsRemoteRequest(void);
void   CkDynCcsSendReply(int len, const void *data);
void  *CkDynCcsDelayReply(void);
void   CkDynCcsSendDelayedReply(void *token, int len, const void *data);

#ifdef __cplusplus
}
#endif

#endif /* CKDYN_H */
