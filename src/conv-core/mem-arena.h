#ifndef _MEM_ARENA_H_
#define _MEM_ARENA_H_

#define USE_BTREE                 1

#if USE_BTREE

/* b-tree definitions */
#define TREE_NODE_SIZE 128 /* a power of 2 is probably best  */
#define TREE_NODE_MID  63  /* must be ceiling(TREE_NODE_SIZE / 2) - 1  */

/* linked list definitions  */
#define LIST_ARRAY_SIZE 64

/* doubly-linked list node */
struct _dllnode {
  struct _dllnode   *previous;
  struct _slotblock *sb;
  struct _dllnode   *next;
};

/* slotblock */
struct _slotblock {
  CmiInt8 startslot;
  CmiInt8 nslots;
  struct _dllnode *listblock;
};

typedef struct _dllnode   dllnode;
typedef struct _slotblock slotblock;

/* b-tree node */
struct _btreenode {
  int num_blocks;
  slotblock blocks[TREE_NODE_SIZE];
  struct _btreenode *child[TREE_NODE_SIZE + 1];
};
typedef struct _btreenode btreenode;

/* slotset */
typedef struct _slotset {
  btreenode *btree_root;
  dllnode *list_array[LIST_ARRAY_SIZE];
} slotset;

#else

typedef struct _slotblock
{
  CmiInt8 startslot;
  CmiInt8 nslots;
} slotblock;

typedef struct _slotset
{
  int maxbuf;
  slotblock *buf;
  CmiInt8 emptyslots;
} slotset;

#endif

slotset *new_slotset(CmiInt8 startslot, CmiInt8 nslots);
CmiInt8 get_slots(slotset *ss, CmiInt8 nslots);
void grab_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots);
void free_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots);

#endif
