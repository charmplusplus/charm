/**************************************************************************
Isomalloc:
A way to allocate memory at the same address on every processor.
This enables linked data structures, like thread stacks, to be migrated
to the same address range on other processors.  This is similar to an
explicitly managed shared memory system.

The memory is used and released via the mmap()/mumap() calls, so unused
memory does not take any (RAM, swap or disk) space.

The way it's implemented is that each processor claims some section 
of the available virtual address space, and satisfies all new allocations
from that space.  Migrating structures use whatever space they started with.

The b-tree implementation has two data structures that are maintained
simultaneously and in conjunction with each other: a b-tree of nodes
containing slotblocks used for quickly finding a particular slotblock;
and an array of doubly-linked lists containing the same slotblocks,
ordered according to the number of free slots in each slotblock, which
is used for quickly finding a block of free slots of a desired size.
The slotset contains pointers to both structures.
print_btree_top_down() and print_list_array() are very useful
functions for viewing the current structure of the tree and lists.

Each doubly-linked list has slotblocks containing between 2^(n-1)+1
and 2^n free slots, where n is the array index (i.e., bin number).
For example, list_array[0] points to a double-linked list of
slotblocks with 1 free slot, list_array[1] points to a double-linked
list of slotblocks with 2 free slots, list_array[2] to slotblocks with
3-4 free slots, list_array[3] to slotblocks with 5-8 free slots, etc.

Written for migratable threads by Milind Bhandarkar around August 2000;
generalized by Orion Lawlor November 2001.  B-tree implementation
added by Ryan Mokos in July 2008.
 *************************************************************************/

#include "converse.h"
#include "memory-isomalloc.h"

#define ISOMALLOC_DEBUG 0

/* 0: use the old isomalloc implementation (array)
1: use the new isomalloc implementation (b-tree)  */
#define USE_BTREE_ISOMALLOC 1

/* b-tree definitions */
#define TREE_NODE_SIZE 128 /* a power of 2 is probably best  */
#define TREE_NODE_MID  63  /* must be cieling(TREE_NODE_SIZE / 2) - 1  */

/* linked list definitions  */
#define LIST_ARRAY_SIZE 64

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h> /* just so I can find dynamically-linked symbols */
#include <unistd.h>

#if CMK_HAS_ADDR_NO_RANDOMIZE
#include <sys/personality.h>
#endif

#if CMK_USE_MEMPOOL_ISOMALLOC
#include "mempool.h"
extern int cutOffPoints[cutOffNum];
#endif 

static int _sync_iso = 0;
#if __FAULT__
static int _restart = 0;
#endif
static int _mmap_probe = 0;

static int read_randomflag(void)
{
  FILE *fp;
  int random_flag;
  fp = fopen("/proc/sys/kernel/randomize_va_space", "r");
  if (fp != NULL) {
    fscanf(fp, "%d", &random_flag);
    fclose(fp);
    if(random_flag) random_flag = 1;
#if CMK_HAS_ADDR_NO_RANDOMIZE
    if(random_flag)
    {
      int persona = personality(0xffffffff);
      if(persona & ADDR_NO_RANDOMIZE)
        random_flag = 0;
    }
#endif
  }
  else {
    random_flag = -1;
  }
  return random_flag;
}

struct CmiIsomallocBlock {
  CmiInt8 slot;   /* First mapped slot */
  CmiInt8 length; /* Length of (user portion of) mapping, in bytes*/
};
typedef struct CmiIsomallocBlock CmiIsomallocBlock;

/* Convert a heap block pointer to/from a CmiIsomallocBlock header */
static void *block2pointer(CmiIsomallocBlock *blockHeader) {
  return (void *)(blockHeader+1);
}
static CmiIsomallocBlock *pointer2block(void *heapBlock) {
  return ((CmiIsomallocBlock *)heapBlock)-1;
}

/* Integral type to be used for pointer arithmetic: */
typedef size_t memRange_t;

/* Size in bytes of a single slot */
static size_t slotsize;
static size_t regionSize;

/* Total number of slots per processor */
static CmiInt8 numslots=0;

/* Start and end of isomalloc-managed addresses.
   If isomallocStart==NULL, isomalloc is disabled.
   */
static char *isomallocStart=NULL;
static char *isomallocEnd=NULL;

/* Utility conversion functions */
static void *slot2addr(CmiInt8 slot) {
  return isomallocStart+((memRange_t)slotsize)*((memRange_t)slot);
}
static int slot2pe(CmiInt8 slot) {
  return (int)(slot/numslots);
}
static CmiInt8 pe2slot(int pe) {
  return pe*numslots;
}
/* Return the number of slots in a block with n user data bytes */
#if CMK_USE_MEMPOOL_ISOMALLOC
static int length2slots(int nBytes) {
  return (nBytes+slotsize-1)/slotsize;
}
#else
static int length2slots(int nBytes) {
  return (sizeof(CmiIsomallocBlock)+nBytes+slotsize-1)/slotsize;
}
#endif
/* ======================================================================
 * New isomalloc (b-tree-based implementation)
 * ====================================================================== */

#if USE_BTREE_ISOMALLOC

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

/* return value for a b-tree insert */
typedef struct _insert_ret_val {
  slotblock sb;
  btreenode *btn;
} insert_ret_val;

/*****************************************************************
 * Find and return the number of the bin (list) to which the number
 * nslots belongs.  The range of each bin is 2^(b-1)+1 to 2^b, where b
 * is the bin number.
 *****************************************************************/

static int find_list_bin(CmiInt8 nslots) {
  int list_bin     = 32;
  CmiInt8 comp_num = 0x100000000LL;
  int inc          = 16;

  while (1) {
    if ((nslots > (comp_num >> 1)) && (nslots <= comp_num)) {
      /* found it */
      return list_bin;
    } else if (nslots < comp_num) {
      /* look left  */
      list_bin -= inc;
      comp_num  = comp_num >> inc;
      if ((inc = inc >> 1) == 0) {
        inc = 1;
      }
    } else {
      /* look right */
      list_bin += inc;
      comp_num  = comp_num << inc;
      if ((inc = inc >> 1) == 0) {
        inc = 1;
      }
    }
  }

}

/*****************************************************************
 * Creates and inserts a new dllnode into list_array (part of the
 * slotset ss) that both points to and is pointed to by the slotblock
 * sb.  This function also returns a pointer to that new dllnode.
 *****************************************************************/

static dllnode *list_insert(slotset *ss, slotblock *sb) {

  /* find the list bin to put the new dllnode in  */
  int list_bin = find_list_bin(sb->nslots);

  /* allocate memory for the new node */
  dllnode *new_dlln = (dllnode *)(malloc_reentrant(sizeof(dllnode)));

  /* insert the dllnode */
  new_dlln->previous = NULL;
  new_dlln->next     = ss->list_array[list_bin];
  new_dlln->sb       = sb;
  if (ss->list_array[list_bin] != NULL) {
    ss->list_array[list_bin]->previous = new_dlln;
  }
  ss->list_array[list_bin] = new_dlln;

  return new_dlln;

}

/*****************************************************************
 * Deletes the dllnode from list_array (part of the slotset ss) that
 * is pointed to by the slotblock sb.
 *****************************************************************/

static void list_delete(slotset *ss, slotblock *sb) {

  /* remove the node from the list */
  if (sb->listblock->next != NULL) {
    sb->listblock->next->previous = sb->listblock->previous;
  }
  if (sb->listblock->previous != NULL) {
    sb->listblock->previous->next = sb->listblock->next;
  } else {  /* first element in the list */
    ss->list_array[find_list_bin(sb->nslots)] = sb->listblock->next;
  }

  /* free the memory from the node */
  free_reentrant(sb->listblock);

}

/*****************************************************************
 * Moves the dllnode dlln to the correct bin (list) of slotset ss
 * based on the number of slots in the slotblock to which dlln points.
 * It is assumed that the slotblock pointed to by dlln has already been
 * updated with the new number of slots.  The integer old_nslots
 * contains the number of slots that used to be in the slotblock.
 *****************************************************************/

static void list_move(slotset *ss, dllnode *dlln, CmiInt8 old_nslots) {

  /* determine the bin the slotblock used to be in */
  int old_bin = find_list_bin(old_nslots);

  /* determine the bin the slotblock is in now */
  int new_bin = find_list_bin(dlln->sb->nslots);

  /* if the old bin and new bin are different, move the slotblock */
  if (new_bin != old_bin) {

    /* remove from old bin */
    if (dlln->previous == NULL) {  /* dlln is the 1st element in the list */
      if (dlln->next != NULL) {
        dlln->next->previous = NULL;
      }
      ss->list_array[old_bin] = dlln->next;
    } else {
      if (dlln->next != NULL) {
        dlln->next->previous = dlln->previous;
      }
      dlln->previous->next = dlln->next;
    }

    /* insert at the beginning of the new bin */
    dlln->next     = ss->list_array[new_bin];
    dlln->previous = NULL;
    if (dlln->next != NULL) {
      dlln->next->previous = dlln;
    }
    ss->list_array[new_bin] = dlln;
  }

}

/*****************************************************************
 * Creates a new b-tree node
 *****************************************************************/

static btreenode *create_btree_node() {
  int i;
  btreenode *btn = (btreenode *)malloc_reentrant(sizeof(btreenode));
  btn->num_blocks = 0;
  for (i = 0; i < TREE_NODE_SIZE; i++) {
    btn->blocks[i].listblock = NULL;
  }
  for (i = 0; i < TREE_NODE_SIZE + 1; i++) {
    btn->child[i] = NULL;
  }
  return btn;
}

/*****************************************************************
 * Find the slotblock in the b-tree that contains startslot.  Returns
 * NULL if such a block cannot be found.
 *****************************************************************/

static slotblock *find_btree_slotblock(btreenode *node, CmiInt8 startslot) {

  /* check if this node exists */
  if ((node == NULL) || (startslot < 0) || (node->num_blocks == 0)) {
    return NULL;
  } else {

    /*** Binary search on this node ***/
    /* initialize */
    int index = node->num_blocks >> 1;
    int inc   = (index >> 1) + (node->num_blocks & 0x1);
    CmiInt8 endslot;

    /* loop until a node is found */
    while (1) {

      /* if startslot is in current slotblock, this is the slotblock */
      endslot = node->blocks[index].startslot + node->blocks[index].nslots - 1;
      if ((startslot >= node->blocks[index].startslot) &&
          (startslot <= endslot)) {
        return &(node->blocks[index]);
      }

      /* else, if startslot is less */
      else if (startslot < node->blocks[index].startslot) {

        /* if this is slotblock 0, take the left child */
        if (index == 0) {
          return find_btree_slotblock(node->child[index], startslot);
        }

        /* else check endslot of the slotblock to the left */
        else {

          /* if startslot > endslot-of-slotblock-to-the-left, take the
             left child */
          endslot = node->blocks[index-1].startslot + 
            node->blocks[index-1].nslots - 1;
          if (startslot > endslot) {
            return find_btree_slotblock(node->child[index], startslot);
          }

          /* else continue to search this node to the left */
          else {
            index -= inc;
            if ((inc = inc >> 1) == 0) {
              inc = 1;
            }
          }
        }
      }

      /* else, startslot must be greater */
      else {

        /* if this is the last slotblock, take the right child */
        if (index == node->num_blocks - 1) {
          return find_btree_slotblock(node->child[index+1], startslot);
        }

        /* else check startslot of the slotblock to the right */
        else {

          /* if startslot < startslot-of-slotblock-to-the-right, then
             take the right child */
          if (startslot < node->blocks[index+1].startslot) {
            return find_btree_slotblock(node->child[index+1], startslot);
          }

          /* else continue to search this node to the right */
          else {
            index += inc;
            if ((inc = inc >> 1) == 0) {
              inc = 1;
            }
          }
        }
      }

    }

  }

}

/*****************************************************************
 * Insert a slotblock into the b-tree starting at startslot and going
 * for nslots slots
 *****************************************************************/

static insert_ret_val btree_insert_int(slotset *ss, btreenode *node, 
    CmiInt8 startslot, CmiInt8 nslots) {

  insert_ret_val irv;

  /*** binary search for the place to insert ***/

  /* initialize */
  int index = node->num_blocks >> 1;
  int inc   = (index >> 1) + (node->num_blocks & 0x1);

  /* loop until an insertion happens */
  while (1) {
    if (startslot < node->blocks[index].startslot) {  /* look to the left */
      if ((index == 0) || 
          (startslot > node->blocks[index-1].startslot)) {
        if (node->child[index] != NULL) {             /* take left child */
          irv = btree_insert_int(ss, node->child[index], startslot, nslots);
          if (irv.btn == NULL) {
            return irv;
          } else {                                    /* merge return value */
            int i, j;                                 /*   insert on left   */
            for (i = node->num_blocks; i > index; i--) {
              node->blocks[i].startslot     = node->blocks[i-1].startslot;
              node->blocks[i].nslots        = node->blocks[i-1].nslots;
              node->blocks[i].listblock     = node->blocks[i-1].listblock;
              node->blocks[i].listblock->sb = &(node->blocks[i]);
              node->child[i+1]              = node->child[i];
            }
            node->blocks[index].startslot     = irv.sb.startslot;
            node->blocks[index].nslots        = irv.sb.nslots;
            node->blocks[index].listblock     = irv.sb.listblock;
            node->blocks[index].listblock->sb = &(node->blocks[index]);
            node->child[index+1]              = irv.btn;
            node->num_blocks++;
            if (node->num_blocks == TREE_NODE_SIZE) {   /* split node */
              btreenode *new_node = create_btree_node();
              for (i = TREE_NODE_MID + 1; i < TREE_NODE_SIZE; i++) {
                j = i - (TREE_NODE_MID + 1);
                new_node->blocks[j].startslot     = node->blocks[i].startslot;
                new_node->blocks[j].nslots        = node->blocks[i].nslots;
                new_node->blocks[j].listblock     = node->blocks[i].listblock;
                new_node->blocks[j].listblock->sb = &(new_node->blocks[j]);
              }
              for (i = TREE_NODE_MID + 1; i <= TREE_NODE_SIZE; i++) {
                new_node->child[i-(TREE_NODE_MID+1)] = node->child[i];
              }
              node->num_blocks     = TREE_NODE_MID;
              new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

              irv.sb.startslot = node->blocks[TREE_NODE_MID].startslot;
              irv.sb.nslots    = node->blocks[TREE_NODE_MID].nslots;
              irv.sb.listblock = node->blocks[TREE_NODE_MID].listblock;
              irv.btn          = new_node;
              return irv;
            } else {
              irv.btn = NULL;
              return irv;
            }
          }
        } else {                                      /* insert to the left */
          int i, j;
          for (i = node->num_blocks; i > index; i--) {
            node->blocks[i].startslot     = node->blocks[i-1].startslot;
            node->blocks[i].nslots        = node->blocks[i-1].nslots;
            node->blocks[i].listblock     = node->blocks[i-1].listblock;
            node->blocks[i].listblock->sb = &(node->blocks[i]);
          }
          node->blocks[index].startslot = startslot;
          node->blocks[index].nslots    = nslots;
          node->blocks[index].listblock = list_insert(ss, &(node->blocks[index]));
          node->num_blocks++;
          if (node->num_blocks == TREE_NODE_SIZE) {   /* split node */
            btreenode *new_node = create_btree_node();
            for (i = TREE_NODE_MID + 1; i < TREE_NODE_SIZE; i++) {
              j = i - (TREE_NODE_MID + 1);
              new_node->blocks[j].startslot     = node->blocks[i].startslot;
              new_node->blocks[j].nslots        = node->blocks[i].nslots;
              new_node->blocks[j].listblock     = node->blocks[i].listblock;
              new_node->blocks[j].listblock->sb = &(new_node->blocks[j]);
            }
            node->num_blocks     = TREE_NODE_MID;
            new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

            irv.sb.startslot = node->blocks[TREE_NODE_MID].startslot;
            irv.sb.nslots    = node->blocks[TREE_NODE_MID].nslots;
            irv.sb.listblock = node->blocks[TREE_NODE_MID].listblock;
            irv.btn          = new_node;
            return irv;
          } else {
            irv.btn = NULL;
            return irv;
          }
        }
      } else {                                        /* search to the left */
        index -= inc;
        if ((inc = inc >> 1) == 0) {
          inc = 1;
        }
      }

    } else {                                          /* look to the right */

      if ((index == node->num_blocks - 1) || 
          (startslot < node->blocks[index+1].startslot)) {
        if (node->child[index+1] != NULL) {           /* take right child */
          irv = btree_insert_int(ss, node->child[index+1], startslot, nslots);
          if (irv.btn == NULL) {
            return irv;
          } else {                                    /* merge return value */
            int i, j;                                 /*   insert on right  */
            for (i = node->num_blocks; i > index + 1; i--) {
              node->blocks[i].startslot     = node->blocks[i-1].startslot;
              node->blocks[i].nslots        = node->blocks[i-1].nslots;
              node->blocks[i].listblock     = node->blocks[i-1].listblock;
              node->blocks[i].listblock->sb = &(node->blocks[i]);
              node->child[i+1]              = node->child[i];
            }
            node->blocks[index+1].startslot     = irv.sb.startslot;
            node->blocks[index+1].nslots        = irv.sb.nslots;
            node->blocks[index+1].listblock     = irv.sb.listblock;
            node->blocks[index+1].listblock->sb = &(node->blocks[index+1]);
            node->child[index+2]                = irv.btn;
            node->num_blocks++;
            if (node->num_blocks == TREE_NODE_SIZE) {   /* split node */
              btreenode *new_node = create_btree_node();
              for (i = TREE_NODE_MID + 1; i < TREE_NODE_SIZE; i++) {
                j = i - (TREE_NODE_MID + 1);
                new_node->blocks[j].startslot     = node->blocks[i].startslot;
                new_node->blocks[j].nslots        = node->blocks[i].nslots;
                new_node->blocks[j].listblock     = node->blocks[i].listblock;
                new_node->blocks[j].listblock->sb = &(new_node->blocks[j]);
              }
              for (i = TREE_NODE_MID + 1; i <= TREE_NODE_SIZE; i++) {
                new_node->child[i-(TREE_NODE_MID+1)] = node->child[i];
              }
              node->num_blocks     = TREE_NODE_MID;
              new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

              irv.sb.startslot = node->blocks[TREE_NODE_MID].startslot;
              irv.sb.nslots    = node->blocks[TREE_NODE_MID].nslots;
              irv.sb.listblock = node->blocks[TREE_NODE_MID].listblock;
              irv.btn          = new_node;
              return irv;
            } else {
              irv.btn = NULL;
              return irv;
            }
          }
        } else {                                      /* insert to the right */
          int i, j;
          for (i = node->num_blocks; i > index + 1; i--) {
            node->blocks[i].startslot     = node->blocks[i-1].startslot;
            node->blocks[i].nslots        = node->blocks[i-1].nslots;
            node->blocks[i].listblock     = node->blocks[i-1].listblock;
            node->blocks[i].listblock->sb = &(node->blocks[i]);
          }
          node->blocks[index+1].startslot = startslot;
          node->blocks[index+1].nslots    = nslots;
          node->blocks[index+1].listblock = list_insert(ss, &(node->blocks[index+1]));
          node->num_blocks++;
          if (node->num_blocks == TREE_NODE_SIZE) {   /* split node */
            btreenode *new_node = create_btree_node();
            for (i = TREE_NODE_MID + 1; i < TREE_NODE_SIZE; i++) {
              j = i - (TREE_NODE_MID + 1);
              new_node->blocks[j].startslot     = node->blocks[i].startslot;
              new_node->blocks[j].nslots        = node->blocks[i].nslots;
              new_node->blocks[j].listblock     = node->blocks[i].listblock;
              new_node->blocks[j].listblock->sb = &(new_node->blocks[j]);
            }
            node->num_blocks = TREE_NODE_MID;
            new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

            irv.sb.startslot = node->blocks[TREE_NODE_MID].startslot;
            irv.sb.nslots    = node->blocks[TREE_NODE_MID].nslots;
            irv.sb.listblock = node->blocks[TREE_NODE_MID].listblock;
            irv.btn          = new_node;
            return irv;
          } else {
            irv.btn = NULL;
            return irv;
          }
        }
      } else {                                        /* search to the right */
        index += inc;
        if ((inc = inc >> 1) == 0) {
          inc = 1;
        }
      }
    }
  }

}

static btreenode *btree_insert(slotset *ss, btreenode *node, 
    CmiInt8 startslot, CmiInt8 nslots) {

  /* check the b-tree root: if it's empty, insert the element in the
     first position */
  if (node->num_blocks == 0) {

    node->num_blocks          = 1;
    node->blocks[0].startslot = startslot;
    node->blocks[0].nslots    = nslots;
    node->blocks[0].listblock = list_insert(ss, &(node->blocks[0]));

  } else {

    /* insert into the b-tree */
    insert_ret_val irv = btree_insert_int(ss, node, startslot, nslots);

    /* if something was returned, we need a new root */
    if (irv.btn != NULL) {
      btreenode *new_root  = create_btree_node();
      new_root->num_blocks = 1;
      new_root->blocks[0].startslot     = irv.sb.startslot;
      new_root->blocks[0].nslots        = irv.sb.nslots;
      new_root->blocks[0].listblock     = irv.sb.listblock;
      new_root->blocks[0].listblock->sb = &(new_root->blocks[0]);
      new_root->child[0] = node;
      new_root->child[1] = irv.btn;
      node = new_root;
    }

  }

  return node;

}

/*****************************************************************
 * Delete the slotblock from the b-tree starting at startslot
 *****************************************************************/

static void btree_delete_int(slotset *ss, btreenode *node, 
    CmiInt8 startslot, slotblock *sb) {

  int i, index, inc;
  int def_child;
  int num_left, num_right, left_pos, right_pos;

  /* If sb is not NULL, we're sending sb down the tree to a leaf to be
     swapped with the next larger startslot so it can be deleted from
     a leaf node (deletions from non-leaf nodes are not allowed
     here).  At this point, the next larger startslot will always be
     found by taking the leftmost child.  */
  if (sb != NULL) {

    if (node->child[0] != NULL) {
      btree_delete_int(ss, node->child[0], startslot, sb);
      index = 0;

    } else {

      /* we're now at a leaf node, so the slotblock can be deleted

         first, copy slotblock 0 to the block passed down (sb) and
         delete the list array node  */
      list_delete(ss, sb);
      sb->startslot     = node->blocks[0].startslot;
      sb->nslots        = node->blocks[0].nslots;
      sb->listblock     = node->blocks[0].listblock;
      sb->listblock->sb = sb;

      /* delete the slotblock */
      for (i = 0; i < (node->num_blocks - 1); i++) {
        node->blocks[i].startslot     = node->blocks[i+1].startslot;
        node->blocks[i].nslots        = node->blocks[i+1].nslots;
        node->blocks[i].listblock     = node->blocks[i+1].listblock;
        node->blocks[i].listblock->sb = &(node->blocks[i]);
      }
      node->num_blocks--;

      return;

    }

  } else {

    /*** Binary search for the slotblock to delete ***/

    /* initialize */
    index = node->num_blocks >> 1;
    inc = (index >> 1) + (node->num_blocks & 0x1);

    /* loop until the slotblock with startslot is found */
    while (1) {

      if (startslot == node->blocks[index].startslot) {   /* found it */
        if (node->child[index+1] != NULL) {               /* not a leaf */
          btree_delete_int(ss, node->child[index+1], 
              startslot, &(node->blocks[index]));
          break;
        } else {                                          /* is a leaf */
          /* delete the slotblock */
          list_delete(ss, &(node->blocks[index]));
          for (i = index; i < (node->num_blocks - 1); i++) {
            node->blocks[i].startslot     = node->blocks[i+1].startslot;
            node->blocks[i].nslots        = node->blocks[i+1].nslots;
            node->blocks[i].listblock     = node->blocks[i+1].listblock;
            node->blocks[i].listblock->sb = &(node->blocks[i]);
          }
          node->num_blocks--;
          return;
        }
      } else {
        if (startslot < node->blocks[index].startslot) {  /* look left */
          if ((index == 0) ||                             /* take left child */
              (startslot > node->blocks[index-1].startslot)) {
            btree_delete_int(ss, node->child[index], startslot, sb);
            break;
          } else {                                        /* search left */
            index -= inc;
            if ((inc = inc >> 1) == 0) {
              inc = 1;
            }
          }
        } else {                                          /* look right */
          if ((index == node->num_blocks - 1) ||          /* take right child */
              (startslot < node->blocks[index+1].startslot)) {
            btree_delete_int(ss, node->child[index+1], startslot, sb);
            break;
          } else {                                        /* search right */
            index += inc;
            if ((inc = inc >> 1) == 0) {
              inc = 1;
            }
          }
        }
      }

    }

  }

  /* At this point, the desired slotblock has been removed, and we're
     going back up the tree.  We must check for deficient nodes that
     require the rotating or combining of elements to maintain a
     balanced b-tree. */
  def_child = -1;

  /* check if one of the child nodes is deficient  */
  if (node->child[index]->num_blocks < TREE_NODE_MID) {
    def_child = index;
  } else if (node->child[index+1]->num_blocks < TREE_NODE_MID) {
    def_child = index + 1;
  }

  if (def_child >= 0) {

    /* if there is a left sibling and it has enough elements, rotate */
    /* to the right */
    if ((def_child != 0) && (node->child[def_child-1] != NULL) && 
        (node->child[def_child-1]->num_blocks > TREE_NODE_MID)) {

      /* move all elements in deficient child to the right */
      for (i = node->child[def_child]->num_blocks; i > 0; i--) {
        node->child[def_child]->blocks[i].startslot = 
          node->child[def_child]->blocks[i-1].startslot;
        node->child[def_child]->blocks[i].nslots = 
          node->child[def_child]->blocks[i-1].nslots;
        node->child[def_child]->blocks[i].listblock = 
          node->child[def_child]->blocks[i-1].listblock;
        node->child[def_child]->blocks[i].listblock->sb = 
          &(node->child[def_child]->blocks[i]);
      }
      for (i = node->child[def_child]->num_blocks + 1; i > 0; i--) {
        node->child[def_child]->child[i] = 
          node->child[def_child]->child[i-1];
      }

      /* move parent element to the deficient child */
      node->child[def_child]->blocks[0].startslot = 
        node->blocks[def_child-1].startslot;
      node->child[def_child]->blocks[0].nslots = 
        node->blocks[def_child-1].nslots;
      node->child[def_child]->blocks[0].listblock = 
        node->blocks[def_child-1].listblock;
      node->child[def_child]->blocks[0].listblock->sb = 
        &(node->child[def_child]->blocks[0]);
      node->child[def_child]->num_blocks++;

      /* move the right-most child of the parent's left child to the
         left-most child of the formerly deficient child  */
      i = node->child[def_child-1]->num_blocks;
      node->child[def_child]->child[0] = 
        node->child[def_child-1]->child[i];

      /* move largest element from left child up to the parent */
      i--;
      node->blocks[def_child-1].startslot = 
        node->child[def_child-1]->blocks[i].startslot;
      node->blocks[def_child-1].nslots = 
        node->child[def_child-1]->blocks[i].nslots;
      node->blocks[def_child-1].listblock = 
        node->child[def_child-1]->blocks[i].listblock;
      node->blocks[def_child-1].listblock->sb = 
        &(node->blocks[def_child-1]);
      node->child[def_child-1]->num_blocks--;

    }

    /* otherwise, if there is a right sibling and it has enough */
    /* elements, rotate to the left */
    else if (((def_child + 1) <= node->num_blocks) && 
        (node->child[def_child+1] != NULL) && 
        (node->child[def_child+1]->num_blocks > TREE_NODE_MID)) {

      /* move parent element to the deficient child */
      i = node->child[def_child]->num_blocks;
      node->child[def_child]->blocks[i].startslot = 
        node->blocks[def_child].startslot;
      node->child[def_child]->blocks[i].nslots = 
        node->blocks[def_child].nslots;
      node->child[def_child]->blocks[i].listblock = 
        node->blocks[def_child].listblock;
      node->child[def_child]->blocks[i].listblock->sb = 
        &(node->child[def_child]->blocks[i]);
      node->child[def_child]->num_blocks++;

      /* move the left-most child of the parent's right child to the
         right-most child of the formerly deficient child  */
      i++;
      node->child[def_child]->child[i] = 
        node->child[def_child+1]->child[0];

      /* move smallest element from right child up to the parent */
      node->blocks[def_child].startslot = 
        node->child[def_child+1]->blocks[0].startslot;
      node->blocks[def_child].nslots = 
        node->child[def_child+1]->blocks[0].nslots;
      node->blocks[def_child].listblock = 
        node->child[def_child+1]->blocks[0].listblock;
      node->blocks[def_child].listblock->sb = 
        &(node->blocks[def_child]);
      node->child[def_child+1]->num_blocks--;

      /* move all elements in the parent's right child to the left  */
      for (i = 0; i < node->child[def_child+1]->num_blocks; i++) {
        node->child[def_child+1]->blocks[i].startslot = 
          node->child[def_child+1]->blocks[i+1].startslot;
        node->child[def_child+1]->blocks[i].nslots = 
          node->child[def_child+1]->blocks[i+1].nslots;
        node->child[def_child+1]->blocks[i].listblock = 
          node->child[def_child+1]->blocks[i+1].listblock;
        node->child[def_child+1]->blocks[i].listblock->sb = 
          &(node->child[def_child+1]->blocks[i]);
      }
      for (i = 0; i < node->child[def_child+1]->num_blocks + 1; i++) {
        node->child[def_child+1]->child[i] = 
          node->child[def_child+1]->child[i+1];
      }
    }

    /* otherwise, merge the deficient node, parent, and the parent's
       other child (one of the deficient node's siblings) by dropping
       the parent down to the level of the children */
    else {

      /* move the parent element into the left child node */
      i = node->child[index]->num_blocks;
      node->child[index]->blocks[i].startslot = 
        node->blocks[index].startslot;
      node->child[index]->blocks[i].nslots = 
        node->blocks[index].nslots;
      node->child[index]->blocks[i].listblock = 
        node->blocks[index].listblock;
      node->child[index]->blocks[i].listblock->sb = 
        &(node->child[index]->blocks[i]);
      node->child[index]->num_blocks++;

      /* move the elements and children of the right child node to the */
      /* left child node */
      num_left  = node->child[index]->num_blocks;
      num_right = node->child[index+1]->num_blocks;
      left_pos;
      right_pos = 0;
      for (left_pos = num_left; left_pos < num_left + num_right; left_pos++) {
        node->child[index]->blocks[left_pos].startslot = 
          node->child[index+1]->blocks[right_pos].startslot;
        node->child[index]->blocks[left_pos].nslots = 
          node->child[index+1]->blocks[right_pos].nslots;
        node->child[index]->blocks[left_pos].listblock = 
          node->child[index+1]->blocks[right_pos].listblock;
        node->child[index]->blocks[left_pos].listblock->sb = 
          &(node->child[index]->blocks[left_pos]);
        right_pos++;
      }
      right_pos = 0;
      for (left_pos = num_left; left_pos < num_left + num_right + 1; left_pos++) {
        node->child[index]->child[left_pos] = 
          node->child[index+1]->child[right_pos];
        right_pos++;
      }
      node->child[index]->num_blocks = num_left + num_right;

      /* delete the right child node */
      free_reentrant(node->child[index+1]);
      node->child[index+1] = NULL;

      /* update the parent node */
      node->num_blocks--;
      for (i = index; i < node->num_blocks; i++) {
        node->blocks[i].startslot     = node->blocks[i+1].startslot;
        node->blocks[i].nslots        = node->blocks[i+1].nslots;
        node->blocks[i].listblock     = node->blocks[i+1].listblock;
        node->blocks[i].listblock->sb = &(node->blocks[i]);
        node->child[i+1]              = node->child[i+2];
      }

    }

  }

}

static btreenode *btree_delete(slotset *ss, btreenode *node, CmiInt8 startslot) {

  /* delete element from the b-tree */
  btree_delete_int(ss, node, startslot, NULL);

  /* if the root node is empty (from a combine operation on the tree),
     the left-most child of the root becomes the new root, unless the
     left-most child is NULL, in which case we leave the root node
     empty but not NULL */
  if (node->num_blocks == 0) {
    if (node->child[0] != NULL) {
      btreenode *new_root = node->child[0];
      free_reentrant(node);
      node = new_root;
    }
  }

  return node;

}

/*****************************************************************
 * Creates a new slotset with nslots entries, starting with all empty
 * slots.  The slot numbers are [startslot, startslot + nslots - 1].
 *****************************************************************/

static slotset *new_slotset(CmiInt8 startslot, CmiInt8 nslots) {
  int i;
  int list_bin;

  /* CmiPrintf("*** New Isomalloc ***\n"); */

  /* allocate memory for the slotset */
  slotset *ss = (slotset *)(malloc_reentrant(sizeof(slotset)));

  /* allocate memory for the b-tree */
  ss->btree_root = create_btree_node();

  /* initialize the b-tree */
  ss->btree_root->num_blocks          = 1;
  ss->btree_root->blocks[0].startslot = startslot;
  ss->btree_root->blocks[0].nslots    = nslots;

  /* initialize the list array */
  for (i = 0; i < LIST_ARRAY_SIZE; i++) {
    ss->list_array[i] = NULL;
  }
  list_bin = find_list_bin(nslots);
  ss->list_array[list_bin] = (dllnode *)(malloc_reentrant(sizeof(dllnode)));
  ss->list_array[list_bin]->previous = NULL;
  ss->list_array[list_bin]->next = NULL;
  ss->list_array[list_bin]->sb = &(ss->btree_root->blocks[0]);

  ss->btree_root->blocks[0].listblock = ss->list_array[list_bin];

  return ss;

}

/*****************************************************************
 * Finds a slotblock containing at least nslots memory slots and
 * returns the startslot of that slotblock; returns -1 if no such
 * slotblock exists.
 *****************************************************************/

static CmiInt8 get_slots(slotset *ss, CmiInt8 nslots) {

  /* calculate the smallest bin (list) to look in first */
  int start_list = find_list_bin(nslots);

  /* search for a slotblock with enough slots */
  int i;
  dllnode *dlln;
  for (i = start_list; i < LIST_ARRAY_SIZE; i++) {
    dlln = ss->list_array[i];
    while (dlln != NULL) {
      if (dlln->sb->nslots >= nslots) {
        return dlln->sb->startslot;
      }
      dlln = dlln->next;
    }
  }

  /* return -1 if no such slotblock exists */
  return (-1);

}

/*****************************************************************
 * Grab a slotblock with the specified range of blocks (nslots blocks
 * starting at sslot).  This is different from get_slots because
 * grab_slots specifies the slots to be grabbed and actually grabs
 * them (removes them from the set of free slots).  get_slots only
 * finds a set of free slots.
 *****************************************************************/

static void grab_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots) {

  CmiInt8 endslot;
  slotblock *sb = find_btree_slotblock(ss->btree_root, sslot);

  if (sb == NULL) {
    CmiAbort("requested a non-existent slotblock\n");
  } else {

    if (sb->startslot == sslot) {

      /* range is exact range of slotblock - delete block from tree */
      if (sb->nslots == nslots) {
        ss->btree_root = btree_delete(ss, ss->btree_root, sslot);
      }

      /* range is at beginning of slotblock - update block range */
      else {
        CmiInt8 old_nslots = sb->nslots;
        sb->startslot     += nslots;
        sb->nslots        -= nslots;
        list_move(ss, sb->listblock, old_nslots);
      }

    } else {

      /* range is at end of slotblock - update block range */
      endslot = sb->startslot + sb->nslots - 1;
      if (endslot == (sslot + nslots - 1)) {
        CmiInt8 old_nslots = sb->nslots;
        sb->nslots        -= nslots;
        list_move(ss, sb->listblock, old_nslots);
      }

      /* range is in middle of slotblock - update block range with the */
      /* new lower range and insert a block with the new upper range */
      else {
        CmiInt8 old_nslots = sb->nslots;
        sb->nslots         = sslot - sb->startslot;
        list_move(ss, sb->listblock, old_nslots);
        ss->btree_root = btree_insert(ss, ss->btree_root, sslot + nslots, 
            endslot - (sslot + nslots) + 1);
      }

    }

  }

}

/*****************************************************************
 * Frees nslots memory slots starting at sslot by either adding them
 * to one of the slotblocks that exists (if the slot ranges are
 * contiguous) or by creating a new slotblock
 *****************************************************************/

static void free_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots) {

  slotblock *sb_low  = find_btree_slotblock(ss->btree_root, sslot - 1);
  slotblock *sb_high = find_btree_slotblock(ss->btree_root, sslot + nslots);

  if (sb_low == NULL) {
    if (sb_high == NULL) {

      /* there is no adjacent slotblock, so create a new one and */
      /* insert it in the b-tree */
      ss->btree_root = btree_insert(ss, ss->btree_root, sslot, nslots);

    } else {

      /* there is an adjacent slotblock to the right, so update its range */
      CmiInt8 old_nslots = sb_high->nslots;
      sb_high->startslot = sslot;
      sb_high->nslots   += nslots;
      list_move(ss, sb_high->listblock, old_nslots);

    }
  } else {
    if (sb_high == NULL) {

      /* there is an adjacent slotblock to the left, so update its range */
      CmiInt8 old_nslots  = sb_low->nslots;
      sb_low->nslots     += nslots;
      list_move(ss, sb_low->listblock, old_nslots);

    } else {

      /* there are adjacent slotblocks on both sides (i.e., the
         slots to be freed exactly span the gap between 2 slotblocks),
         so update the range of the lower slotblock and delete the
         upper one */
      CmiInt8 old_nslots = sb_low->nslots;
      sb_low->nslots     = sb_low->nslots + nslots + sb_high->nslots;
      list_move(ss, sb_low->listblock, old_nslots);
      ss->btree_root = btree_delete(ss, ss->btree_root, sb_high->startslot);

    }
  }

}

/*****************************************************************
 * Recursively free the allocated memory of the b-tree
 *****************************************************************/

static void delete_btree(btreenode *node) {
  int i;
  for (i = 0; i <= node->num_blocks; i++) {
    if (node->child[i] != NULL) {
      delete_btree(node->child[i]);
      free_reentrant(node->child[i]);
    } else {
      return;
    }
  }
}

/*****************************************************************
 * Free the allocated memory of the list array
 *****************************************************************/

static void delete_list_array(slotset *ss) {
  int i;
  dllnode *dlln;
  for (i = 0; i < LIST_ARRAY_SIZE; i++) {
    dlln = ss->list_array[i];
    if (dlln != NULL) {
      while (dlln->next != NULL) {
        dlln = dlln->next;
      }
      while (dlln->previous != NULL) {
        dlln = dlln->previous;
        free_reentrant(dlln->next);
      }
      free_reentrant(dlln);
    }
  }
}

/*****************************************************************
 * Free the allocated memory of the slotset
 *****************************************************************/

static void delete_slotset(slotset *ss) {
  delete_btree(ss->btree_root);
  delete_list_array(ss);
  free_reentrant(ss->btree_root);
  free_reentrant(ss);
}

/*****************************************************************
 * Print the contents of the b-tree on the screen in a top-down
 * fashion, starting with the root and progressing to the sub-trees
 *****************************************************************/

/* prints the elements in a single b-tree node */
static void print_btree_node(btreenode *node, int node_num) {
  int i;
  CmiPrintf("Node %2d: ", node_num);
  for (i = 0; i < node->num_blocks; i++) {
    CmiPrintf("%d:[%lld,%lld] ", i, node->blocks[i].startslot, node->blocks[i].nslots);
  }
  CmiPrintf("\n");
}

/* returns 1 if there's another level to print; 0 if not */
static int print_btree_level(btreenode *node, int level, int current_level, int node_num) {
  int i, another_level;
  for (i = 0; i <= node->num_blocks; i++) {
    if (current_level == level) {
      print_btree_node(node, node_num);
      if (node->child[0] == NULL) {
        return 0;
      } else {
        return 1;
      }
    } else {
      another_level = print_btree_level(node->child[i], level, 
          current_level + 1, i);
    }
  }
  return another_level;
}

static void print_btree_top_down(btreenode *node) {
  int level = 0;
  int another_level;
  do {
    CmiPrintf("B-tree Level %d\n", level);
    CmiPrintf("---------------\n");
    another_level = print_btree_level(node, level, 0, 0);
    level++;
    CmiPrintf("\n\n");
  } while (another_level);
}

/*****************************************************************
 * Print the contents of the list arry on the screen
 *****************************************************************/

static void print_list_array(slotset *ss) {
  int i;
  dllnode *dlln;
  CmiPrintf("List Array\n");
  CmiPrintf("----------\n");
  for (i = 0; i < LIST_ARRAY_SIZE; i++) {
    CmiPrintf("List %2d: ", i);
    dlln = ss->list_array[i];
    while (dlln != NULL) {
      if (dlln->previous != NULL) {
        CmiPrintf("<->");
      } else {
        CmiPrintf(" ->");
      }
      CmiPrintf("[%lld,%lld]", dlln->sb->startslot, dlln->sb->nslots);
      dlln = dlln->next;
    }
    CmiPrintf("\n");
  }
}

#if ISOMALLOC_DEBUG
static void print_slots(slotset *ss) {
  print_btree_top_down(ss->btree_root);
  print_list_array(ss);
}
#else
#  define print_slots(ss) /*empty*/
#endif



/* ======================================================================
 * Old isomalloc (array-based implementation)
 * ====================================================================== */

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

/*
 * creates a new slotset of nslots entries, starting with all
 * empty slots. The slot numbers are [startslot,startslot+nslot-1]
 */

  static slotset *
new_slotset(CmiInt8 startslot, CmiInt8 nslots)
{
  /* CmiPrintf("*** Old isomalloc ***\n"); */
  int i;
  slotset *ss = (slotset*) malloc_reentrant(sizeof(slotset));
  _MEMCHECK(ss);
  ss->maxbuf = 16;
  ss->buf = (slotblock *) malloc_reentrant(sizeof(slotblock)*ss->maxbuf);
  _MEMCHECK(ss->buf);
  ss->emptyslots = nslots;
  ss->buf[0].startslot = startslot;
  ss->buf[0].nslots = nslots;
  for (i=1; i<ss->maxbuf; i++)
    ss->buf[i].nslots = 0;
  return ss;
}

/*
 * returns new block of empty slots. if it cannot find any, returns (-1).
 */

  static CmiInt8
get_slots(slotset *ss, CmiInt8 nslots)
{
  /* CmiPrintf("old get: nslots=%lld\n", nslots); */
  int i;
  if(ss->emptyslots < nslots)
    return (-1);
  for(i=0;i<(ss->maxbuf);i++)
    if(ss->buf[i].nslots >= nslots)
      return ss->buf[i].startslot;
  return (-1);
}

/* just adds a slotblock to an empty position in the given slotset. */

  static void
add_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots)
{
  int pos; 
  int emptypos = -1;
  if (nslots == 0)
    return;
  for (pos=0; pos < (ss->maxbuf); pos++) {
    if (ss->buf[pos].nslots == 0) {
      emptypos = pos;
      break; /* found empty slotblock */
    }
  }
  if (emptypos == (-1)) /*no empty slotblock found */
  {
    int i;
    int newsize = ss->maxbuf*2;
    slotblock *newbuf = (slotblock *) malloc_reentrant(sizeof(slotblock)*newsize);
    _MEMCHECK(newbuf);
    for (i=0; i<(ss->maxbuf); i++)
      newbuf[i] = ss->buf[i];
    for (i=ss->maxbuf; i<newsize; i++)
      newbuf[i].nslots  = 0;
    free_reentrant(ss->buf);
    ss->buf = newbuf;
    emptypos = ss->maxbuf;
    ss->maxbuf = newsize;
  }
  ss->buf[emptypos].startslot = sslot;
  ss->buf[emptypos].nslots = nslots;
  ss->emptyslots += nslots;
  return;
}

/* grab a slotblock with specified range of blocks
 * this is different from get_slots, since it pre-specifies the
 * slots to be grabbed.
 */
  static void
grab_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots)
{
  /* CmiPrintf("old grab: sslot=%lld nslots=%lld\n", sslot, nslots); */
  CmiInt8 pos, eslot, e;
  eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    if (ss->buf[pos].nslots == 0)
      continue;
    e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if(sslot >= ss->buf[pos].startslot && eslot <= e)
    {
      CmiInt8 old_nslots;
      old_nslots = ss->buf[pos].nslots;
      ss->buf[pos].nslots = sslot - ss->buf[pos].startslot;
      ss->emptyslots -= (old_nslots - ss->buf[pos].nslots);
      add_slots(ss, sslot + nslots, old_nslots - ss->buf[pos].nslots - nslots);
      /* CmiPrintf("grab: sslot=%lld nslots=%lld pos=%lld i=%d\n", sslot, nslots, pos, i); */
      return;
    }
  }
  CmiAbort("requested a non-existent slotblock\n");
}

/*
 * Frees slot by adding it to one of the blocks of empty slots.
 * this slotblock is one which is contiguous with the slots to be freed.
 * if it cannot find such a slotblock, it creates a new slotblock.
 * If the buffer fills up, it adds up extra buffer space.
 */
  static void
free_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots)
{
  /* CmiPrintf("old free: sslot=%lld nslots=%lld\n", sslot, nslots); */
  int pos;
  /* eslot is the ending slot of the block to be freed */
  CmiInt8 eslot = sslot + nslots;
  for (pos=0; pos < (ss->maxbuf); pos++)
  {
    CmiInt8 e = ss->buf[pos].startslot + ss->buf[pos].nslots;
    if (ss->buf[pos].nslots == 0)
      continue;
    /* e is the ending slot of pos'th slotblock */
    if (e == sslot) /* append to the current slotblock */
    {
      ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
      /* CmiPrintf("free:append pos=%d\n", pos); */
      return;
    }
    if(eslot == ss->buf[pos].startslot) /* prepend to the current slotblock */
    {
      ss->buf[pos].startslot = sslot;
      ss->buf[pos].nslots += nslots;
      ss->emptyslots += nslots;
      /* CmiPrintf("free:prepend pos=%d\n", pos); */
      return;
    }
  }
  /* if we are here, it means we could not find a slotblock that the */
  /* block to be freed was combined with. */
  /* CmiPrintf("free: pos=%d\n", pos); */
  add_slots(ss, sslot, nslots);
}

/*
 * destroys slotset-- currently unused
 static void
 delete_slotset(slotset* ss)
 {
 free_reentrant(ss->buf);
 free_reentrant(ss);
 }
 */

#if ISOMALLOC_DEBUG
  static void
print_slots(slotset *ss)
{
  int i;
  CmiPrintf("[%d] maxbuf = %d\n", CmiMyPe(), ss->maxbuf);
  CmiPrintf("[%d] emptyslots = %d\n", CmiMyPe(), ss->emptyslots);
  for(i=0;i<ss->maxbuf;i++) {
    if(ss->buf[i].nslots)
      CmiPrintf("[%d] (start[%d], end[%d], num=%d) \n", CmiMyPe(), ss->buf[i].startslot, 
          ss->buf[i].startslot+ss->buf[i].nslots, ss->buf[i].nslots);
  }
}
#else
/*#  define print_slots(ss) */ /*empty*/
  static void
print_slots(slotset *ss)
{
  int i;
  CmiPrintf("[%d] maxbuf = %d\n", CmiMyPe(), ss->maxbuf);
  CmiPrintf("[%d] emptyslots = %d\n", CmiMyPe(), ss->emptyslots);
  for(i=0;i<ss->maxbuf;i++) {
    if(ss->buf[i].nslots)
      CmiPrintf("[%d] (start[%d], end[%d], num=%d) \n", CmiMyPe(), ss->buf[i].startslot, 
          ss->buf[i].startslot+ss->buf[i].nslots, ss->buf[i].nslots);
  }
}

#endif


#endif

/* ======================= END OLD ISOMALLOC ============================ */


/*This version of the allocate/deallocate calls are used if the 
  real mmap versions are disabled.*/
static int disabled_map_warned=0;
static void *disabled_map(int nBytes) 
{
  if (!disabled_map_warned) {
    disabled_map_warned=1;
    if (CmiMyPe()==0)
      CmiError("charm isomalloc.c> Warning: since mmap() doesn't work,"
          " you won't be able to migrate threads\n");
  }
  return malloc(nBytes);
}
static void disabled_unmap(void *bk) {
  free(bk);
}

/*Turn off isomalloc memory, for the given reason*/
static void disable_isomalloc(const char *why)
{
  isomallocStart=NULL;
  isomallocEnd=NULL;
  if (CmiMyPe() == 0)
    CmiPrintf("[%d] isomalloc.c> Disabling isomalloc because %s\n",CmiMyPe(),why);
}

#if ! CMK_HAS_MMAP
/****************** Manipulate memory map (Win32 non-version) *****************/
static void *call_mmap_fixed(void *addr,size_t len) {
  CmiAbort("isomalloc.c: mmap_fixed should never be called here.");
  return NULL;
}
static void *call_mmap_anywhere(size_t len) {
  CmiAbort("isomalloc.c: mmap_anywhere should never be called here.");
  return NULL;
}
static void call_munmap(void *addr,size_t len) {
  CmiAbort("isomalloc.c: munmap should never be called here.");
}

  static int 
init_map(char **argv)
{
  return 0; /*Isomalloc never works without mmap*/
}
#else /* CMK_HAS_MMAP */
/****************** Manipulate memory map (UNIX version) *****************/
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#if !CMK_HAS_MMAP_ANON
CpvStaticDeclare(int, zerofd); /*File descriptor for /dev/zero, for mmap*/
#endif

/**
 * Maps this address with these flags.
 */
static void *call_mmap(void *addr,size_t len, int flags) {
  void *ret=mmap(addr, len, PROT_READ|PROT_WRITE,
#if CMK_HAS_MMAP_ANON
      flags|MAP_PRIVATE|MAP_ANON,-1,
#else
      flags|MAP_PRIVATE,CpvAccess(zerofd),
#endif
      0);
  if (ret==((void*)(-1))) return (void *)0; /* all-ones means failure */
  else return ret;
}
static void *call_mmap_fixed(void *addr,size_t len) {
  return call_mmap(addr,len,MAP_FIXED);
}
static void *call_mmap_anywhere(size_t len) {
  return call_mmap((void *)0,len,0);
}

/* Unmaps this address range */
static void call_munmap(void *addr,size_t len) {
  int retval;
  if (addr == 0) return; /* NULL address is never mapped */ 
  retval = munmap(addr, len);
  if (retval==(-1))
    CmiAbort("munmap call failed to deallocate requested memory.\n");
}

  static int 
init_map(char **argv)
{
#if CMK_HAS_MMAP_ANON
  /*Don't need /dev/zero*/
#else
  CpvInitialize(int, zerofd);  
  CpvAccess(zerofd) = open("/dev/zero", O_RDWR);
  if(CpvAccess(zerofd)<0)
    return 0; /* Cannot open /dev/zero or use MMAP_ANON, so can't mmap memory */
#endif
  return 1;
}

#endif /* UNIX memory map */

/**
 * maps the virtual memory associated with slot using mmap
 */
  static CmiIsomallocBlock *
map_slots(CmiInt8 slot, CmiInt8 nslots)
{
  void *pa;
  void *addr=slot2addr(slot);
  pa = call_mmap_fixed(addr, slotsize*nslots);

  if (pa==NULL)
  { /*Map just failed completely*/
#if ISOMALLOC_DEBUG
    perror("mmap failed");
    CmiPrintf("[%d] tried to mmap %p, but encountered error\n",CmiMyPe(),addr);
#endif
    return NULL;
  }
  if (pa != addr)
  { /*Map worked, but gave us back the wrong place*/
#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] tried to mmap %p, but got %p back\n",CmiMyPe(),addr,pa);
#endif
    call_munmap(addr,slotsize*nslots);
    return NULL;
  }
#if ISOMALLOC_DEBUG
  CmiPrintf("[%d] mmap'd slots %ld-%ld to address %p\n",CmiMyPe(),
      slot,slot+nslots-1,addr);
#endif
  return (CmiIsomallocBlock *)pa;
}

/*
 * unmaps the virtual memory associated with slot using munmap
 */
  static void
unmap_slots(CmiInt8 slot, CmiInt8 nslots)
{
  void *addr=slot2addr(slot);
  call_munmap(addr, slotsize*nslots);
#if ISOMALLOC_DEBUG
  CmiPrintf("[%d] munmap'd slots %ld-%ld from address %p\n",CmiMyPe(),
      slot,slot+nslots-1,addr);
#endif
}

static void map_failed(CmiInt8 s,CmiInt8 n)
{
  void *addr=slot2addr(s);
  CmiError("charm isomalloc.c> map failed to allocate %d bytes at %p, errno:%d.\n",
      slotsize*n, addr, errno);
  CmiAbort("Exiting\n");
}



/************ Address space voodoo: find free address range **********/

CpvStaticDeclare(slotset *, myss); /*My managed slots*/

#if CMK_USE_MEMPOOL_ISOMALLOC
CtvDeclare(mempool_type *, threadpool); /*Thread managed pools*/

//alloc function to be used by mempool
void * isomallocfn (size_t *size, mem_handle_t *mem_hndl, int expand_flag)
{
  CmiInt8 s,n,i;
  void *newaddr;
  n=length2slots(*size);
  /*Always satisfy mallocs with local slots:*/
  s=get_slots(CpvAccess(myss),n);
  if (s==-1) {
    CmiError("Not enough address space left on processor %d to isomalloc %d bytes!\n",
              CmiMyPe(),*size);
    CmiAbort("Out of virtual address space for isomalloc");
  }
  grab_slots(CpvAccess(myss),s,n);
  for (i=0; i<5; i++) {
    newaddr=map_slots(s,n);
    if (newaddr!=NULL) break;
#if CMK_HAS_USLEEP
    if (errno == ENOMEM) { usleep(rand()%1000); continue; }
    else break;
#endif
  }
  if (!newaddr) map_failed(s,n);
  *((CmiInt8 *)mem_hndl) = s;
  *size = n*slotsize;
  return newaddr;
}

//free function to be used by mempool
void isofreefn(void *ptr, mem_handle_t mem_hndl)
{
  call_munmap(ptr, ((block_header *)ptr)->size);
}
#endif

/*This struct describes a range of virtual addresses*/
typedef struct {
  char *start; /*First byte of region*/
  memRange_t len; /*Number of bytes in region*/
  const char *type; /*String describing memory in region (debugging only)*/
} memRegion_t;

/*Estimate the top of the current stack*/
static void *__cur_stack_frame(void)
{
  char __dummy = 'A';
  void *top_of_stack=(void *)&__dummy;
  return top_of_stack;
}
/*Estimate the location of the static data region*/
static void *__static_data_loc(void)
{
  static char __dummy;
  return (void *)&__dummy;
}

/*Pointer comparison is in these subroutines, because
  comparing arbitrary pointers is nonportable and tricky.
  */
static int pointer_lt(const char *a,const char *b) {
  return ((memRange_t)a)<((memRange_t)b);
}
static int pointer_ge(const char *a,const char *b) {
  return ((memRange_t)a)>=((memRange_t)b);
}

static char *pmin(char *a,char *b) {return pointer_lt(a,b)?a:b;}
static char *pmax(char *a,char *b) {return pointer_lt(a,b)?b:a;}

const static memRange_t meg=1024u*1024u; /*One megabyte*/
const static memRange_t gig=1024u*1024u*1024u; /*One gigabyte*/

/*Check if this memory location is usable.  
  If not, return 1.
  */
static int bad_location(char *loc) {
  void *addr;
  addr=call_mmap_fixed(loc,slotsize);
  if (addr==NULL || addr!=loc) {
#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] Skipping unmappable space at %p\n",CmiMyPe(),loc);
#endif
    return 1; /*No good*/
  }
  call_munmap(addr,slotsize);
  return 0; /*This works*/
}

/* Split this range up into n pieces, returning the size of each piece */
static memRange_t divide_range(memRange_t len,int n) {
  return (len+1)/n;
}

/* Return if this memory region has *any* good parts. */
static int partially_good(char *start,memRange_t len,int n) {
  int i;
  memRange_t quant=divide_range(len,n);
  CmiAssert (quant > 0);
  for (i=0;i<n;i++)
    if (!bad_location(start+i*quant)) return 1; /* it's got some good parts */
  return 0; /* all locations are bad */
}

/* Return if this memory region is usable at n samples.  
*/
static int good_range(char *start,memRange_t len,int n) {
  int i;
  memRange_t quant=divide_range(len,n);
#if ISOMALLOC_DEBUG
  CmiPrintf("good_range: %lld, %d\n", quant, n);
#endif
  CmiAssert (quant > 0);

  for (i=0;i<n;i++)
    if (bad_location(start+i*quant)) return 0; /* it's got some bad parts */
  /* It's all good: */
  return 1;
}

/*Check if this entire memory range, or some subset 
  of the range, is usable.  If so, write it into max.
  */
static void check_range(char *start,char *end,memRegion_t *max)
{
  memRange_t len;
  CmiUInt8 tb = (CmiUInt8)gig*1024ul;   /* One terabyte */
  CmiUInt8 vm_limit = tb*256ul;   /* terabyte */

  if (start>=end) return; /*Ran out of hole*/
  len=(memRange_t)end-(memRange_t)start;

#if 0
  /* too conservative */
  if (len/gig>64u) { /* This is an absurd amount of space-- cut it down, for safety */
    start+=16u*gig;
    end=start+32u*gig;
    len=(memRange_t)end-(memRange_t)start;  
  }
#else
  /* Note: 256TB == 248 bytes.  So a 48-bit virtual-address CPU 
   *    can only actually address 256TB of space. */
  if (len/tb>10u) { /* This is an absurd amount of space-- cut it down, for safety */
    const memRange_t other_libs=16ul*gig; /* space for other libraries to use */
    start+=other_libs;
    end=pmin(start+vm_limit-2*other_libs, end-other_libs);
    len=(memRange_t)end-(memRange_t)start;
  }
#endif
  if (len<=max->len) return; /*It's too short already!*/
#if ISOMALLOC_DEBUG
  CmiPrintf("[%d] Checking at %p - %p\n",CmiMyPe(),start,end);
#endif

  /* Check the middle of the range */
  if (!good_range(start,len,256)) {
    /* Try to split into subranges: */
    int i,n=2;
#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] Trying to split bad address space at %p - %p...\n",CmiMyPe(),start,end);
#endif
    len=divide_range(len,n);
    for (i=0;i<n;i++) {
      char *cur=start+i*len;
      if (partially_good(cur,len,16))
        check_range(cur,cur+len,max);
    }
    return; /* Hopefully one of the subranges will be any good */
  }
  else /* range is good */
  { 
#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] Address space at %p - %p is largest\n",CmiMyPe(),start,end);
#endif

    /*If we got here, we're the new largest usable range*/
    max->len=len;
    max->start=start;
    max->type="Unused";
  }
}

/*Find the first available memory region of at least the
  given size not touching any data in the used list.
  */
static memRegion_t find_free_region(memRegion_t *used,int nUsed,int atLeast) 
{
  memRegion_t max;
  int i,j;  

  max.start=0; 
  max.len=atLeast;
  /*Find the largest hole between regions*/
  for (i=0;i<nUsed;i++) {
    /*Consider a hole starting at the end of region i*/
    char *holeStart=used[i].start+used[i].len;
    char *holeEnd=(void *)(-1);

    /*Shrink the hole by all others*/ 
    for (j=0;j<nUsed && pointer_lt(holeStart,holeEnd);j++) {
      if (pointer_lt(used[j].start,holeStart)) 
        holeStart=pmax(holeStart,used[j].start+used[j].len);
      else if (pointer_lt(used[j].start,holeEnd)) 
        holeEnd=pmin(holeEnd,used[j].start);
    } 

    check_range(holeStart,holeEnd,&max);
  }

  return max; 
}

/*
   By looking at the address range carefully, try to find 
   the largest usable free region on the machine.
   */
static int find_largest_free_region(memRegion_t *destRegion) {
  char *staticData =(char *) __static_data_loc();
  char *code = (char *)&find_free_region;
  char *threadData = (char *)&errno;
  char *codeDll = (char *)fprintf;
  char *heapLil = (char*) malloc(1);
  char *heapBig = (char*) malloc(6*meg);
  char *stack = (char *)__cur_stack_frame();
  size_t mmapAnyLen = 1*meg;
  char *mmapAny = (char*) call_mmap_anywhere(mmapAnyLen);

  int i,nRegions=0;
  memRegion_t regions[10]; /*used portions of address space*/
  memRegion_t freeRegion; /*Largest unused block of address space*/

  /*Mark off regions of virtual address space as ususable*/
  regions[nRegions].type="NULL";
  regions[nRegions].start=NULL; 
#if CMK_POWER7 && CMK_64BIT
  regions[nRegions++].len=2u*gig;   /* on bluedrop, don't mess with the lower memory region */
#else
  regions[nRegions++].len=16u*meg;
#endif

  regions[nRegions].type="Static program data";
  regions[nRegions].start=staticData; regions[nRegions++].len=256u*meg;

  regions[nRegions].type="Program executable code";
  regions[nRegions].start=code; regions[nRegions++].len=256u*meg;

  regions[nRegions].type="Heap (small blocks)";
  regions[nRegions].start=heapLil; regions[nRegions++].len=1u*gig;

  regions[nRegions].type="Heap (large blocks)";
  regions[nRegions].start=heapBig; regions[nRegions++].len=1u*gig;

  regions[nRegions].type="Stack space";
  regions[nRegions].start=stack; regions[nRegions++].len=256u*meg;

  regions[nRegions].type="Program dynamically linked code";
  regions[nRegions].start=codeDll; regions[nRegions++].len=256u*meg; 

  regions[nRegions].type="Result of a non-fixed call to mmap";
  regions[nRegions].start=mmapAny; regions[nRegions++].len=2u*gig; 

  regions[nRegions].type="Thread private data";
  regions[nRegions].start=threadData; regions[nRegions++].len=256u*meg; 

  _MEMCHECK(heapBig); free(heapBig);
  _MEMCHECK(heapLil); free(heapLil); 
  call_munmap(mmapAny,mmapAnyLen);

  /*Align each memory region*/
  for (i=0;i<nRegions;i++) {
    memRegion_t old=regions[i];
    memRange_t p=(memRange_t)regions[i].start;
    p&=~(regions[i].len-1); /*Round start down to a len-boundary (mask off low bits)*/
    regions[i].start=(char *)p;
#if CMK_MACOSX
    if (regions[i].start+regions[i].len*2>regions[i].start) regions[i].len *= 2;
#endif
#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] Memory map: %p - %p (len: %lu => %lu) %s \n",CmiMyPe(),
        regions[i].start,regions[i].start+regions[i].len,
        old.len, regions[i].len, regions[i].type);
#endif
  }

  /*Find a large, unused region in this map: */
  freeRegion=find_free_region(regions,nRegions,(512u)*meg);

  if (freeRegion.start==0) 
  { /*No free address space-- disable isomalloc:*/
    return 0;
  }
  else /* freeRegion is valid */
  {
    *destRegion=freeRegion;

    return 1;
  }
}

static int try_largest_mmap_region(memRegion_t *destRegion)
{
  void *bad_alloc=(void*)(-1); /* mmap error return address */
  void *range, *good_range=NULL;
  double shrink = 1.5;
  static int count = 0;
  size_t size=((size_t)(-1l)), good_size=0;
  int retry = 0;
  if (sizeof(size_t) >= 8) size = size>>2;  /* 25% of machine address space! */
  while (1) { /* test out an allocation of this size */
#if CMK_HAS_MMAP
    range=mmap(NULL,size,PROT_READ|PROT_WRITE,
        MAP_PRIVATE
#if CMK_HAS_MMAP_ANON
        |MAP_ANON
#endif
#if CMK_HAS_MMAP_NORESERVE
        |MAP_NORESERVE
#endif
        ,-1,0);
#else
    range = bad_alloc;
#endif
    if (range == bad_alloc) {  /* mmap failed */
#if ISOMALLOC_DEBUG
      /* CmiPrintf("[%d] test failed at size: %llu error: %d\n", CmiMyPe(), size, errno);  */
#endif
#if CMK_HAS_USLEEP
      if (retry++ < 5) { usleep(rand()%10000); continue; }
      else retry = 0;
#endif
      size=(double)size/shrink; /* shrink request */
      if (size<=0) return 0; /* mmap doesn't work */
    }
    else { /* this allocation size is available */
#if ISOMALLOC_DEBUG
      CmiPrintf("[%d] available: %p, %lld\n", CmiMyPe(), range, size);
#endif
      call_munmap(range,size); /* needed/wanted? */
      if (size > good_size) {
        good_range = range;
        good_size = size;
        size=((double)size)*1.1;
        continue;
      }
      break;
    }
  }
  CmiAssert(good_range!=NULL);
  destRegion->start=good_range; 
  destRegion->len=good_size;
#if ISOMALLOC_DEBUG
  pid_t pid = getpid();
  {
    char s[128];
    sprintf(s, "cat /proc/%d/maps", pid);
    system(s);
  }
  CmiPrintf("[%d] try_largest_mmap_region: %p, %lld\n", CmiMyPe(), good_range, good_size);
#endif
  return 1;
}

#ifndef CMK_CPV_IS_SMP
#define CMK_CPV_IS_SMP
#endif

static void init_ranges(char **argv)
{
  memRegion_t freeRegion;
  /*Largest value a signed int can hold*/
  memRange_t intMax=(((memRange_t)1)<<(sizeof(int)*8-1))-1;
  int pagesize = 0;
  if (CmiMyRank()==0 && numslots==0)
  { /* Find the largest unused region of virtual address space */
    /*Round slot size up to nearest page size*/
#if CMK_USE_MEMPOOL_ISOMALLOC
    slotsize=1024*1024;
#else
    slotsize=16*1024;
#endif 
#if CMK_HAS_GETPAGESIZE
    pagesize = getpagesize();
#endif
    if (pagesize < CMK_MEMORY_PAGESIZE)
      pagesize = CMK_MEMORY_PAGESIZE;
    slotsize=(slotsize+pagesize-1) & ~(pagesize-1);

#if ISOMALLOC_DEBUG
    if (CmiMyPe() == 0)
      CmiPrintf("[%d] Using slotsize of %d\n", CmiMyPe(), slotsize);
#endif
    freeRegion.len=0u;

#ifdef CMK_MMAP_START_ADDRESS /* Hardcoded start address, for machines where automatic fails */
    freeRegion.start=CMK_MMAP_START_ADDRESS;
    freeRegion.len=CMK_MMAP_LENGTH_MEGS*meg;
#endif

    if (freeRegion.len==0u)  {
      if (_mmap_probe == 1) {
        if (try_largest_mmap_region(&freeRegion)) _sync_iso = 1;
      }
      else {
        if (freeRegion.len==0u) find_largest_free_region(&freeRegion);
      }
    }

#if 0
    /*Make sure our largest slot number doesn't overflow an int:*/
    if (freeRegion.len/slotsize>intMax)
      freeRegion.len=intMax*slotsize;
#endif

    if (freeRegion.len==0u) {
      disable_isomalloc("no free virtual address space");
    }
    else /* freeRegion.len>0, so can isomalloc */
    {
#if ISOMALLOC_DEBUG
      CmiPrintf("[%d] Isomalloc memory region: %p - %p (%d megs)\n",CmiMyPe(),
          freeRegion.start,freeRegion.start+freeRegion.len,
          freeRegion.len/meg);
#endif
    }
  }             /* end if myrank == 0 */

  CmiNodeAllBarrier();

  /*
     on some machines, isomalloc memory regions on different nodes 
     can be different. use +isomalloc_sync to calculate the
     intersect of all memory regions on all nodes.
     */
  if (_sync_iso == 1)
  {
#ifdef __FAULT__
        if(_restart == 1){
            CmiUInt8 s = (CmiUInt8)freeRegion.start;
            CmiUInt8 e = (CmiUInt8)(freeRegion.start+freeRegion.len);
            CmiUInt8 ss, ee;
            int try_count, fd;
            char fname[128];
            sprintf(fname,".isomalloc");
            try_count = 0;
            while ((fd = open(fname, O_RDONLY)) == -1 && try_count<10000){
                try_count++;
            }
            if (fd == -1) {
                CmiAbort("isomalloc_sync failed during restart, make sure you have a shared file system.");
            }
            read(fd, &ss, sizeof(CmiUInt8));
            read(fd, &ee, sizeof(CmiUInt8));
            close(fd);
            if (ss < s || ee > e)
                CmiAbort("isomalloc_sync failed during restart, virtual memory regions do not overlap.");
            else {
                freeRegion.start = (void *)ss;
                freeRegion.len = (char *)ee -(char *)ss;
            }
            CmiPrintf("[%d] consolidated Isomalloc memory region at restart: %p - %p (%d megs)\n",CmiMyPe(),freeRegion.start,freeRegion.start+freeRegion.len,freeRegion.len/meg);
            goto AFTER_SYNC;
        }
#endif
    if (CmiMyRank() == 0 && freeRegion.len > 0u) {
      if (CmiBarrier() == -1 && CmiMyPe()==0) 
        CmiAbort("Charm++ Error> +isomalloc_sync requires CmiBarrier() implemented.\n");
      else {
	/* cppcheck-suppress uninitStructMember */
        CmiUInt8 s = (CmiUInt8)freeRegion.start;
	/* cppcheck-suppress uninitStructMember */
        CmiUInt8 e = (CmiUInt8)(freeRegion.start+freeRegion.len);
        int fd, i;
        char fname[128];

        if (CmiMyNode()==0) CmiPrintf("Charm++> synchronizing isomalloc memory region...\n");

        sprintf(fname,".isomalloc.%d", CmiMyNode());

        /* remove file before writing for safe */
        unlink(fname);
#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        system("sync");
#endif

        CmiBarrier();

        /* write region into file */
        while ((fd = open(fname, O_WRONLY|O_TRUNC|O_CREAT, 0644)) == -1) 
#ifndef __MINGW_H
          CMK_CPV_IS_SMP
#endif
            ;
        write(fd, &s, sizeof(CmiUInt8));
        write(fd, &e, sizeof(CmiUInt8));
        close(fd);

#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        system("sync");
#endif

        CmiBarrier();

        for (i=0; i<CmiNumNodes(); i++) {
          CmiUInt8 ss, ee; 
          int try_count;
          char fname[128];
          if (i==CmiMyNode()) continue;
          sprintf(fname,".isomalloc.%d", i);
          try_count = 0;
          while ((fd = open(fname, O_RDONLY)) == -1 && try_count<10000)
          {
            try_count++;
#ifndef __MINGW_H
            CMK_CPV_IS_SMP
#endif
              ;
          }
          if (fd == -1) {
            CmiAbort("isomalloc_sync failed, make sure you have a shared file system.");
          }
          read(fd, &ss, sizeof(CmiUInt8));
          read(fd, &ee, sizeof(CmiUInt8));
#if ISOMALLOC_DEBUG
          if (CmiMyPe() == 0) CmiPrintf("[%d] load node %d isomalloc region: %lx %lx. \n",
              CmiMyPe(), i, ss, ee);
#endif
          close(fd);
          if (ss>s) s = ss;
          if (ee<e) e = ee;
        }

        CmiBarrier();

        unlink(fname);
#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        system("sync");
#endif

        /* update */
        if (s > e)  {
          if (CmiMyPe()==0) CmiPrintf("[%d] Invalid isomalloc region: %lx - %lx.\n", CmiMyPe(), s, e);
          CmiAbort("isomalloc> failed to find consolidated isomalloc region!");
        }
        freeRegion.start = (void *)s;
        freeRegion.len = (char *)e -(char *)s;

        if (CmiMyPe() == 0)
          CmiPrintf("[%d] consolidated Isomalloc memory region: %p - %p (%d megs)\n",CmiMyPe(),
              freeRegion.start,freeRegion.start+freeRegion.len,
              freeRegion.len/meg);
#if __FAULT__
                if(CmiMyPe() == 0){
                    int fd;
                    char fname[128];
                    CmiUInt8 s = (CmiUInt8)freeRegion.start;
                    CmiUInt8 e = (CmiUInt8)(freeRegion.start+freeRegion.len);
                    sprintf(fname,".isomalloc");
                    while ((fd = open(fname, O_WRONLY|O_TRUNC|O_CREAT, 0644)) == -1);
                    write(fd, &s, sizeof(CmiUInt8));
                    write(fd, &e, sizeof(CmiUInt8));
                    close(fd);
                }
#endif
      }   /* end of barrier test */
    } /* end of rank 0 */
    else {
      CmiBarrier();
      CmiBarrier();
      CmiBarrier();
      CmiBarrier();
    }
  }

#ifdef __FAULT__
    AFTER_SYNC:
#endif

  if (CmiMyRank() == 0 && freeRegion.len > 0u)
  {
    /*Isomalloc covers entire unused region*/
    isomallocStart=freeRegion.start;
    isomallocEnd=freeRegion.start+freeRegion.len;
    numslots=(freeRegion.len/slotsize)/CmiNumPes();

#if ISOMALLOC_DEBUG
    CmiPrintf("[%d] Can isomalloc up to %lu megs per pe\n",CmiMyPe(),
        ((memRange_t)numslots)*slotsize/meg);
#endif
  }

  /*SMP Mode: wait here for rank 0 to initialize numslots before calculating myss*/
  CmiNodeAllBarrier(); 

  CpvInitialize(slotset *, myss);
  CpvAccess(myss) = NULL;

#if CMK_USE_MEMPOOL_ISOMALLOC
  CmiLock(_smp_mutex);
  CtvInitialize(mempool_type *, threadpool);
  CtvAccess(threadpool) = NULL;
  CmiUnlock(_smp_mutex);
#endif

  if (isomallocStart!=NULL) {
    CpvAccess(myss) = new_slotset(pe2slot(CmiMyPe()), numslots);
  }
}


/************* Communication: for grabbing/freeing remote slots *********/
typedef struct _slotmsg
{
  char cmicore[CmiMsgHeaderSizeBytes];
  int pe; /*Source processor*/
  CmiInt8 slot; /*First requested slot*/
  CmiInt8 nslots; /*Number of requested slots*/
} slotmsg;

static slotmsg *prepare_slotmsg(CmiInt8 slot,CmiInt8 nslots)
{
  slotmsg *m=(slotmsg *)CmiAlloc(sizeof(slotmsg));
  m->pe=CmiMyPe();
  m->slot=slot;
  m->nslots=nslots;
  return m;
}

static void grab_remote(slotmsg *msg)
{
  grab_slots(CpvAccess(myss),msg->slot,msg->nslots);
  CmiFree(msg);
}

static void free_remote(slotmsg *msg)
{
  free_slots(CpvAccess(myss),msg->slot,msg->nslots);
  CmiFree(msg);
}
static int grab_remote_idx, free_remote_idx;

struct slotOP {
  /*Function pointer to perform local operation*/
  void (*local)(slotset *ss,CmiInt8 s,CmiInt8 n);
  /*Index to perform remote operation*/
  int remote;
};
typedef struct slotOP slotOP;
static slotOP grabOP,freeOP;

static void init_comm(char **argv)
{
  grab_remote_idx=CmiRegisterHandler((CmiHandler)grab_remote);
  free_remote_idx=CmiRegisterHandler((CmiHandler)free_remote);	
  grabOP.local=grab_slots;
  grabOP.remote=grab_remote_idx;
  freeOP.local=free_slots;
  freeOP.remote=free_remote_idx;	
}

/*Apply the given operation to the given slots which
  lie on the given processor.*/
static void one_slotOP(const slotOP *op,int pe,CmiInt8 s,CmiInt8 n)
{
  /*Shrink range to only those covered by this processor*/
  /*First and last slot for this processor*/
  CmiInt8 p_s=pe2slot(pe), p_e=pe2slot(pe+1);
  CmiInt8 e=s+n;
  if (s<p_s) s=p_s;
  if (e>p_e) e=p_e;
  n=e-s;

  /*Send off range*/
  if (pe==CmiMyPe()) 
    op->local(CpvAccess(myss),s,n);
  else 
  {/*Remote request*/
    slotmsg *m=prepare_slotmsg(s,n);
    CmiSetHandler(m, freeOP.remote);
    CmiSyncSendAndFree(pe,sizeof(slotmsg),m);
  }
}

/*Apply the given operation to all slots in the range [s, s+n) 
  After a restart from checkpoint, a slotset can cross an 
  arbitrary set of processors.
  */
static void all_slotOP(const slotOP *op,CmiInt8 s,CmiInt8 n)
{
  int spe=slot2pe(s), epe=slot2pe(s+n-1);
  int pe;
  for (pe=spe; pe<=epe; pe++)
    one_slotOP(op,pe,s,n);
}

/************** External interface ***************/
#if CMK_USE_MEMPOOL_ISOMALLOC
void *CmiIsomalloc(int size, CthThread tid)
{
  CmiInt8 s,n,i;
  CmiIsomallocBlock *blk;
  if (isomallocStart==NULL) return disabled_map(size);
  if(tid != NULL) {
    if(CtvAccessOther(tid,threadpool) == NULL) {
#if ISOMALLOC_DEBUG
      printf("Init Mempool in %d for %d\n",CthSelf(), tid);
#endif
      CtvAccessOther(tid,threadpool) = mempool_init(2*(size+sizeof(CmiIsomallocBlock)+sizeof(mempool_header))+sizeof(mempool_type), isomallocfn, isofreefn,0);
    }
    blk = (CmiIsomallocBlock*)mempool_malloc(CtvAccessOther(tid,threadpool),size+sizeof(CmiIsomallocBlock),1);
  } else {
    if(CtvAccess(threadpool) == NULL) {
#if ISOMALLOC_DEBUG
      printf("Init Mempool in %d\n",CthSelf());
#endif
      CtvAccess(threadpool) = mempool_init(2*(size+sizeof(CmiIsomallocBlock)+sizeof(mempool_header))+sizeof(mempool_type), isomallocfn, isofreefn,0);
    }
    blk = (CmiIsomallocBlock*)mempool_malloc(CtvAccess(threadpool),size+sizeof(CmiIsomallocBlock),1);
  }
  blk->slot=(CmiInt8)blk;
  blk->length=size;
  return block2pointer(blk);
}
#else
void *CmiIsomalloc(int size, CthThread tid)
{
  CmiInt8 s,n,i;
  CmiIsomallocBlock *blk;
  if (isomallocStart==NULL) return disabled_map(size);
  n=length2slots(size);
  /*Always satisfy mallocs with local slots:*/
  s=get_slots(CpvAccess(myss),n);
  if (s==-1) {
    CmiError("Not enough address space left on processor %d to isomalloc %d bytes!\n",
        CmiMyPe(),size);
    CmiAbort("Out of virtual address space for isomalloc");
  }
  grab_slots(CpvAccess(myss),s,n);
  for (i=0; i<5; i++) {
    blk=map_slots(s,n);
    if (blk!=NULL) break;
#if CMK_HAS_USLEEP
    if (errno == ENOMEM) { usleep(rand()%1000); continue; }
    else break;
#endif
  }
  if (!blk) map_failed(s,n);
  blk->slot=s;
  blk->length=size;
  return block2pointer(blk);
}
#endif

#define MALLOC_ALIGNMENT           ALIGN_BYTES
#define MINSIZE                    (sizeof(CmiIsomallocBlock))

/** return an aligned isomalloc memory, the alignment occurs after the
 *  first 'reserved' bytes.  Total requested size is (size+reserved)
 */
static void *_isomallocAlign(size_t align, size_t size, size_t reserved, CthThread t) {
  void *ptr;
  CmiIntPtr ptr2align;
  CmiInt8 s;

  if (align < MINSIZE) align = MINSIZE;
  /* make sure alignment is power of 2 */
  if ((align & (align - 1)) != 0) {
    size_t a = MALLOC_ALIGNMENT * 2;
    while ((unsigned long)a < (unsigned long)align) a <<= 1;
    align = a;
  }
  s = size + reserved + align;
  ptr = CmiIsomalloc(s,t);
  ptr2align = (CmiIntPtr)ptr;
  ptr2align += reserved;
  if (ptr2align % align != 0) { /* misaligned */
    CmiIsomallocBlock *blk = pointer2block(ptr);  /* save block */
    CmiIsomallocBlock savedblk = *blk;
    ptr2align = (ptr2align + align - 1) & -((CmiInt8) align);
    ptr2align -= reserved;
    ptr = (void*)ptr2align;
    blk = pointer2block(ptr);      /* restore block */
    *blk = savedblk;
  }
  return ptr;
}

void *CmiIsomallocAlign(size_t align, size_t size, CthThread t)
{
  return _isomallocAlign(align, size, 0, t);
}

int CmiIsomallocEnabled()
{
  return (isomallocStart!=NULL);
}

void CmiIsomallocPup(pup_er p,void **blockPtrPtr)
{
  CmiIsomallocBlock *blk;
  CmiInt8 s,length;
  CmiInt8 n;
#if CMK_USE_MEMPOOL_ISOMALLOC
  CmiAbort("Incorrect pup is called\n");
#endif
  if (isomallocStart==NULL) CmiAbort("isomalloc is disabled-- cannot use IsomallocPup");

  if (!pup_isUnpacking(p)) 
  { /*We have an existing block-- unpack start slot & length*/
    blk=pointer2block(*blockPtrPtr);
    s=blk->slot;
    length=blk->length;
  }

  pup_int8(p,&s);
  pup_int8(p,&length);
  n=length2slots(length);

  if (pup_isUnpacking(p)) 
  { /*Must allocate a new block in its old location*/
    if (pup_isUserlevel(p) || pup_isRestarting(p))
    {	/*Checkpoint: must grab old slots (even remote!)*/
      all_slotOP(&grabOP,s,n);
    }
    blk=map_slots(s,n);
    if (!blk) map_failed(s,n);
    blk->slot=s;
    blk->length=length;
    *blockPtrPtr=block2pointer(blk);
  }

  /*Pup the allocated data*/
  pup_bytes(p,*blockPtrPtr,length);

  if (pup_isDeleting(p)) 
  { /*Unmap old slots, but do not mark as free*/
    unmap_slots(s,n);
    *blockPtrPtr=NULL; /*Zero out user's pointer*/
  }
}

void CmiIsomallocFree(void *blockPtr)
{
  if (isomallocStart==NULL) {
    disabled_unmap(blockPtr);
  }
  else if (blockPtr!=NULL)
  {
#if CMK_USE_MEMPOOL_ISOMALLOC
    mempool_free_thread((void*)pointer2block(blockPtr)->slot);
#else
    CmiIsomallocBlock *blk=pointer2block(blockPtr);
    CmiInt8 s=blk->slot; 
    CmiInt8 n=length2slots(blk->length);
    unmap_slots(s,n);
    /*Mark used slots as free*/
    all_slotOP(&freeOP,s,n);
#endif
  }
}

CmiInt8  CmiIsomallocLength(void *block)
{
  return pointer2block(block)->length;
}

/*Return true if this address is in the region managed by isomalloc*/
int CmiIsomallocInRange(void *addr)
{
  if (isomallocStart==NULL) return 0; /* There is no range we manage! */
  return (addr == NULL) || (pointer_ge((char *)addr,isomallocStart) && 
    pointer_lt((char*)addr,isomallocEnd));
}

void CmiIsomallocInit(char **argv)
{
#if CMK_NO_ISO_MALLOC
  disable_isomalloc("isomalloc disabled by conv-mach");
#else
  if (CmiGetArgFlagDesc(argv,"+noisomalloc","disable isomalloc")) {
    disable_isomalloc("isomalloc disabled by user.");
    return;
  }
#if CMK_MMAP_PROBE
  _mmap_probe = 1;
#elif CMK_MMAP_TEST
  _mmap_probe = 0;
#endif
  if (CmiGetArgFlagDesc(argv,"+isomalloc_probe","call mmap to probe the largest available isomalloc region"))
    _mmap_probe = 1;
  if (CmiGetArgFlagDesc(argv,"+isomalloc_test","mmap test common areas for the largest available isomalloc region"))
    _mmap_probe = 0;
  if (CmiGetArgFlagDesc(argv,"+isomalloc_sync","synchronize isomalloc region globaly"))
    _sync_iso = 1;
#if __FAULT__
  if (CmiGetArgFlagDesc(argv,"+restartisomalloc","restarting isomalloc on this processor after a crash"))
    _restart = 1;
#endif
  init_comm(argv);
  if (!init_map(argv)) {
    disable_isomalloc("mmap() does not work");
  }
  else {
    if (CmiMyPe() == 0) {
      int _sync_iso_warned = 0;
      if (read_randomflag() == 1) {    /* randomization stack pointer */
        if (_sync_iso == 1)
          printf("Warning> Randomization of stack pointer is turned on in kernel.\n");
        else {
          _sync_iso_warned = 1;
          printf("Warning> Randomization of stack pointer is turned on in kernel, thread migration may not work! Run 'echo 0 > /proc/sys/kernel/randomize_va_space' as root to disable it, or try run with '+isomalloc_sync'.  \n");
        }
      }
#if CMK_SMP
      if (_sync_iso == 0 && _sync_iso_warned == 0)
        printf("Warning> using Isomalloc in SMP mode, you may need to run with '+isomalloc_sync'.\n");
#endif
    }
    init_ranges(argv);
  }
#endif
}

/***************** BlockList interface *********
  This was moved here from memory-isomalloc.c when it 
  was realized that a list-of-isomalloc'd-blocks is useful for
  more than just isomalloc heaps.
  */

/*Convert a slot to a user address*/
static char *Slot_toUser(CmiIsomallocBlockList *s) {return (char *)(s+1);}
static CmiIsomallocBlockList *Slot_fmUser(void *s) {return ((CmiIsomallocBlockList *)s)-1;}

/*Build a new blockList.*/
CmiIsomallocBlockList *CmiIsomallocBlockListNew(CthThread tid)
{
  CmiIsomallocBlockList *ret;
  ret=(CmiIsomallocBlockList *)CmiIsomalloc(sizeof(*ret),tid);
  ret->next=ret; /*1-entry circular linked list*/
  ret->prev=ret;
  return ret;
}

/* BIGSIM_OOC DEBUGGING */
static void print_myslots();

/*Pup all the blocks in this list.  This amounts to two circular
  list traversals.  Because everything's isomalloc'd, we don't even
  have to restore the pointers-- they'll be restored automatically!
  */
#if CMK_USE_MEMPOOL_ISOMALLOC
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **lp, CthThread tid)
{
  mempool_type *mptr;
  block_header *current, *block_head;
  slot_header *currSlot;
  void *newblock;
  CmiInt8 slot;
  size_t size;
  int flags[2];
  int i, j;
  int dopup = 1;
  int numBlocks = 0, numSlots = 0, flag = 1;

  if(!pup_isUnpacking(p)) {
    if(CtvAccessOther(tid,threadpool) == NULL) {
      dopup = 0;
    } else {
      dopup = 1;
    }
  }
  
  pup_int(p,&dopup);
  if(!dopup)  return;

#if ISOMALLOC_DEBUG
  printf("[%d] My rank is %lld Pupping for %lld with isUnpack %d isDelete %d \n",CmiMyPe(),CthSelf(),tid,pup_isUnpacking(p),pup_isDeleting(p));
#endif
  flags[0] = 0; flags[1] = 1;
  if(!pup_isUnpacking(p)) {
    mptr = CtvAccessOther(tid,threadpool);
    current = MEMPOOL_GetBlockHead(mptr);
    while(current != NULL) {
      numBlocks++;
      current = MEMPOOL_GetBlockNext(current)?(block_header *)((char*)mptr+MEMPOOL_GetBlockNext(current)):NULL;
    }
#if ISOMALLOC_DEBUG
    printf("Number of blocks packed %d\n",numBlocks);
#endif
    pup_int(p,&numBlocks);
    current = MEMPOOL_GetBlockHead(mptr);
    while(current != NULL) {
      pup_bytes(p,&(MEMPOOL_GetBlockSize(current)),sizeof(MEMPOOL_GetBlockSize(current)));
      pup_bytes(p,&(MEMPOOL_GetBlockMemHndl(current)),sizeof(CmiInt8));
      numSlots = 0;
      if(flag) {
        pup_bytes(p,current,sizeof(mempool_type));
        currSlot = (slot_header*)((char*)current+sizeof(mempool_type));
      } else {
        pup_bytes(p,current,sizeof(block_header));
        currSlot = (slot_header*)((char*)current+sizeof(block_header));
      }
      while(currSlot != NULL) {
        numSlots++;
        currSlot = (MEMPOOL_GetSlotGNext(currSlot))?(slot_header*)((char*)mptr+MEMPOOL_GetSlotGNext(currSlot)):NULL;
      }
      pup_int(p,&numSlots);
      if(flag) {
        currSlot = (slot_header*)((char*)current+sizeof(mempool_type));
        flag = 0;
      } else {
        currSlot = (slot_header*)((char*)current+sizeof(block_header));
      }
      while(currSlot != NULL) {
        pup_int(p,&cutOffPoints[currSlot->size]);
        if(MEMPOOL_GetSlotStatus(currSlot)) {
          pup_int(p,&flags[0]);
          pup_bytes(p,(void*)currSlot,sizeof(slot_header));
        } else {
          pup_int(p,&flags[1]);
          pup_bytes(p,(void*)currSlot,MEMPOOL_GetSlotSize(currSlot));
        }
        currSlot = (MEMPOOL_GetSlotGNext(currSlot))?(slot_header*)((char*)mptr+MEMPOOL_GetSlotGNext(currSlot)):NULL;
      }
      current = (MEMPOOL_GetBlockNext(current))?(block_header *)((char*)mptr+MEMPOOL_GetBlockNext(current)):NULL;
    }
  }

  if(pup_isUnpacking(p)) {
    pup_int(p,&numBlocks);
#if ISOMALLOC_DEBUG
    printf("Number of blocks to be unpacked %d\n",numBlocks);
#endif
    for(i = 0; i < numBlocks; i++) { 
      pup_bytes(p,&size,sizeof(size));
      pup_bytes(p,&slot,sizeof(slot));
      newblock = map_slots(slot,size/slotsize);
      if(flag) {
        mptr = (mempool_type*)newblock;
        pup_bytes(p,newblock,sizeof(mempool_type));
        newblock = (char*)newblock + sizeof(mempool_type);
        flag = 0;
      } else {
        pup_bytes(p,newblock,sizeof(block_header));
        newblock = (char*)newblock + sizeof(block_header);
      }
      pup_int(p,&numSlots);
      for(j=0; j < numSlots; j++) {
        pup_int(p,&flags[0]);
        pup_int(p,&flags[1]);
        if(flags[1] == 0) {
          pup_bytes(p,newblock,sizeof(slot_header));
        } else {
          pup_bytes(p,newblock,flags[0]);
        }
        newblock = (char*)newblock + flags[0];
      }
    }
#if CMK_USE_MEMPOOL_ISOMALLOC || (CMK_SMP && CMK_CONVERSE_UGNI)
    mptr->mempoolLock = CmiCreateLock();
#endif  
  }
  pup_bytes(p,lp,sizeof(int*));
  if(pup_isDeleting(p)) {
    *lp=NULL;
  }
}
#else
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **lp, CthThread tid)
{
  /* BIGSIM_OOC DEBUGGING */
  /* if(!pup_isUnpacking(p)) print_myslots(); */

  int i,nBlocks=0;
  CmiIsomallocBlockList *cur=NULL, *start=*lp;

  /*Count the number of blocks in the list*/
  if (!pup_isUnpacking(p)) {
    nBlocks=1; /*<- Since we have to skip the start block*/
    for (cur=start->next; cur!=start; cur=cur->next) 
      nBlocks++;
    /*Prepare for next trip around list:*/
    cur=start;
  }
  pup_int(p,&nBlocks);

  /*Pup each block in the list*/
  for (i=0;i<nBlocks;i++) {
    void *newBlock=cur;
    if (!pup_isUnpacking(p)) 
    { /*While packing, we traverse the list to find our blocks*/
      cur=cur->next;
    }
    CmiIsomallocPup(p,&newBlock);
    if (i==0 && pup_isUnpacking(p))
      *lp=(CmiIsomallocBlockList *)newBlock;
  }
  if (pup_isDeleting(p))
    *lp=NULL;

  /* BIGSIM_OOC DEBUGGING */
  /* if(pup_isUnpacking(p)) print_myslots(); */
}
#endif

/*Delete all the blocks in this list.*/
void CmiIsomallocBlockListDelete(CmiIsomallocBlockList *l)
{
  CmiIsomallocBlockList *start=l;
  CmiIsomallocBlockList *cur=start;
  if (cur==NULL) return; /*Already deleted*/
  do {
    CmiIsomallocBlockList *doomed=cur;
    cur=cur->next; /*Have to stash next before deleting cur*/
    CmiIsomallocFree(doomed);
  } while (cur!=start);
}

/*Allocate a block from this blockList*/
void *CmiIsomallocBlockListMalloc(CmiIsomallocBlockList *l,size_t nBytes)
{
  CmiIsomallocBlockList *n; /*Newly created slot*/
  n=(CmiIsomallocBlockList *)CmiIsomalloc(sizeof(CmiIsomallocBlockList)+nBytes,NULL);
  /*Link the new block into the circular blocklist*/
  n->prev=l;
  n->next=l->next;
  l->next->prev=n;
  l->next=n;
  return Slot_toUser(n);
}

/*Allocate a block from this blockList with alighment */
void *CmiIsomallocBlockListMallocAlign(CmiIsomallocBlockList *l,size_t align,size_t nBytes)
{
  CmiIsomallocBlockList *n; /*Newly created slot*/
  n=(CmiIsomallocBlockList *)_isomallocAlign(align,nBytes,sizeof(CmiIsomallocBlockList),NULL);
  /*Link the new block into the circular blocklist*/
  n->prev=l;
  n->next=l->next;
  l->next->prev=n;
  l->next=n;
  return Slot_toUser(n);
}

/*Remove this block from its list and memory*/
void CmiIsomallocBlockListFree(void *block)
{
  CmiIsomallocBlockList *n=Slot_fmUser(block);
#if DOHEAPCHECK
  if (n->prev->next!=n || n->next->prev!=n) 
    CmiAbort("Heap corruption detected in isomalloc block list header!\n"
        "  Run with ++debug and look for writes to negative array indices");
#endif
  /*Link ourselves out of the blocklist*/
  n->prev->next=n->next;
  n->next->prev=n->prev;
  CmiIsomallocFree(n);
}

/* BIGSIM_OOC DEBUGGING */
static void print_myslots(){
  CmiPrintf("[%d] my slot set=%p\n", CmiMyPe(), CpvAccess(myss));
  print_slots(CpvAccess(myss));
}



