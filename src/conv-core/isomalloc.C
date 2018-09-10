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

#include "pup.h"
#include "pup_stl.h"
#include "converse.h"
#include "memory-isomalloc.h"

#define ISOMALLOC_DEBUG 0

#if ISOMALLOC_DEBUG
#define DEBUG_PRINT(...) CmiPrintf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#define ISOMEMPOOL_DEBUG 0

#if ISOMEMPOOL_DEBUG
#define IMP_DBG(...) CmiPrintf(__VA_ARGS__)
#else
#define IMP_DBG(...)
#endif

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

#ifdef _WIN32
# include <io.h>
# define open _open
# define close _close
# define read _read
# define write _write
#endif

#if CMK_HAS_ADDR_NO_RANDOMIZE
#include <sys/personality.h>
#endif

#if CMK_USE_MEMPOOL_ISOMALLOC && !USE_ISOMEMPOOL
#include "mempool.h"
extern int cutOffPoints[cutOffNum];
#endif 

int _sync_iso = 0;
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
    if (fscanf(fp, "%d", &random_flag) != 1) {
      CmiAbort("Isomalloc> fscanf failed reading /proc/sys/kernel/randomize_va_space!");
    }
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

/*
 * "User" here means:
 * 1. Not including CmiIsomallocBlock
 * 2. Including CmiIsomallocBlockList for memory-isomalloc allocations
 */
struct CmiIsomallocBlock {
#if !CMK_USE_MEMPOOL_ISOMALLOC && !USE_ISOMEMPOOL
  CmiInt8 slot;   /* First mapped slot */
  CmiInt8 length; /* Length of (user portion of) mapping, in bytes*/
  CmiInt8 align;  /* User-requested alignment */
  CmiInt8 alignoffset; /* User-requested position that is aligned */
#else
  void *start; /* What the underlying allocator actually returned */
  CmiInt8 length;
#endif
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
static CmiInt8 addr2slot(const void *addr) {
  return (CmiInt8)(uintptr_t)((const char *)addr - isomallocStart) / slotsize;
}
static int slot2pe(CmiInt8 slot) {
  return (int)(slot/numslots);
}
static CmiInt8 pe2slot(int pe) {
  return pe*numslots;
}
/* Return the number of slots in a block with n user data bytes */
static size_t length2slots(size_t nBytes) {
  return (nBytes+slotsize-1)/slotsize;
}

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

static int find_list_bin(CmiInt8 nslots)
{
  int list_bin     = 32;
  CmiInt8 comp_num = 0x100000000LL;
  int inc          = 16;

  while (1)
  {
    if ((comp_num >> 1) < nslots && nslots <= comp_num)
    {
      /* found it */
      return list_bin;
    }
    else if (nslots < comp_num)
    {
      /* look left  */
      list_bin -= inc;
      comp_num >>= inc;
      if ((inc >>= 1) == 0)
        inc = 1;
    }
    else
    {
      /* look right */
      list_bin += inc;
      comp_num <<= inc;
      if ((inc >>= 1) == 0)
        inc = 1;
    }
  }
}

/*****************************************************************
 * Creates and inserts a new dllnode into list_array (part of the
 * slotset ss) that both points to and is pointed to by the slotblock
 * sb.  This function also returns a pointer to that new dllnode.
 *****************************************************************/

static dllnode *list_insert(slotset *ss, slotblock *sb)
{
  /* find the list bin to put the new dllnode in  */
  int list_bin = find_list_bin(sb->nslots);

  /* allocate memory for the new node */
  auto new_dlln = (dllnode *)malloc_reentrant(sizeof(dllnode));

  /* insert the dllnode */
  dllnode *& bin = ss->list_array[list_bin];
  new_dlln->previous = NULL;
  new_dlln->next     = bin;
  new_dlln->sb       = sb;
  if (bin != NULL)
    bin->previous = new_dlln;
  bin = new_dlln;

  return new_dlln;
}

/*****************************************************************
 * Deletes the dllnode from list_array (part of the slotset ss) that
 * is pointed to by the slotblock sb.
 *****************************************************************/

static void list_delete(slotset *ss, slotblock *sb)
{
  dllnode * listblock = sb->listblock;

  /* remove the node from the list */
  if (listblock->next != NULL)
    listblock->next->previous = listblock->previous;

  if (listblock->previous != NULL)
    listblock->previous->next = listblock->next;
  else /* first element in the list */
    ss->list_array[find_list_bin(sb->nslots)] = listblock->next;

  /* free the memory from the node */
  free_reentrant(listblock);
}

/*****************************************************************
 * Moves the dllnode dlln to the correct bin (list) of slotset ss
 * based on the number of slots in the slotblock to which dlln points.
 * It is assumed that the slotblock pointed to by dlln has already been
 * updated with the new number of slots.  The integer old_nslots
 * contains the number of slots that used to be in the slotblock.
 *****************************************************************/

static void list_move(slotset *ss, dllnode *dlln, CmiInt8 old_nslots)
{
  /* determine the bin the slotblock used to be in */
  int old_bin = find_list_bin(old_nslots);

  /* determine the bin the slotblock is in now */
  int new_bin = find_list_bin(dlln->sb->nslots);

  /* if the old bin and new bin are different, move the slotblock */
  if (new_bin != old_bin)
  {
    if (dlln->next != NULL)
      dlln->next->previous = dlln->previous;

    /* remove from old bin */
    if (dlln->previous == NULL) /* dlln is the 1st element in the list */
      ss->list_array[old_bin] = dlln->next;
    else
      dlln->previous->next = dlln->next;

    /* insert at the beginning of the new bin */
    dlln->next     = ss->list_array[new_bin];
    dlln->previous = NULL;
    if (dlln->next != NULL)
      dlln->next->previous = dlln;

    ss->list_array[new_bin] = dlln;
  }
}

/*****************************************************************
 * Creates a new b-tree node
 *****************************************************************/

static btreenode *create_btree_node()
{
  auto btn = (btreenode *)malloc_reentrant(sizeof(btreenode));
  btn->num_blocks = 0;

  for (slotblock * b = btn->blocks, * const b_end = b + TREE_NODE_SIZE; b < b_end; ++b)
    b->listblock = NULL;

  for (btreenode ** c = btn->child, ** const c_end = c + TREE_NODE_SIZE + 1; c < c_end; ++c)
    *c = NULL;

  return btn;
}

/*****************************************************************
 * Find the slotblock in the b-tree that contains startslot.  Returns
 * NULL if such a block cannot be found.
 *****************************************************************/

static slotblock *find_btree_slotblock(btreenode *node, CmiInt8 startslot)
{
  /* check if this node exists */
  if (node == NULL || startslot < 0 || node->num_blocks == 0)
  {
    return NULL;
  }
  else
  {
    /*** Binary search on this node ***/

    /* initialize */
    int index = node->num_blocks >> 1;
    int inc   = (index >> 1) + (node->num_blocks & 0x1);

    /* loop until a node is found */
    while (1)
    {
      slotblock & block = node->blocks[index];
      const CmiInt8 endslot = block.startslot + block.nslots - 1;

      /* if startslot is in current slotblock, this is the slotblock */
      if (block.startslot <= startslot && startslot <= endslot)
      {
        return &block;
      }
      else if (startslot < block.startslot) /* if startslot is less */
      {
        /* if this is slotblock 0, take the left child */
        if (index == 0)
        {
          return find_btree_slotblock(node->child[index], startslot);
        }
        else /* check endslot of the slotblock to the left */
        {
          slotblock & leftblock = node->blocks[index-1];
          const CmiInt8 leftblock_endslot = leftblock.startslot + leftblock.nslots - 1;

          /* if startslot > endslot-of-slotblock-to-the-left,
             take the left child */
          if (startslot > leftblock_endslot)
          {
            return find_btree_slotblock(node->child[index], startslot);
          }
          else /* continue to search this node to the left */
          {
            index -= inc;
            if ((inc >>= 1) == 0)
              inc = 1;
          }
        }
      }
      else /* startslot must be greater */
      {
        /* if this is the last slotblock, take the right child */
        if (index == node->num_blocks-1)
        {
          return find_btree_slotblock(node->child[index+1], startslot);
        }
        else /* check startslot of the slotblock to the right */
        {
          /* if startslot < startslot-of-slotblock-to-the-right,
             take the right child */
          if (startslot < node->blocks[index+1].startslot)
          {
            return find_btree_slotblock(node->child[index+1], startslot);
          }
          else /* continue to search this node to the right */
          {
            index += inc;
            if ((inc >>= 1) == 0)
              inc = 1;
          }
        }
      }
    }
  }
}

static void copy_slotblock(slotblock & dst, const slotblock & src)
{
  dst = src;
  dst.listblock->sb = &dst;
}

static void insert_slotblock(slotblock & block, slotset *ss, CmiInt8 startslot, CmiInt8 nslots)
{
  block.startslot = startslot;
  block.nslots    = nslots;
  block.listblock = list_insert(ss, &block);
}

/*****************************************************************
 * Insert a slotblock into the b-tree starting at startslot and going
 * for nslots slots
 *****************************************************************/

static insert_ret_val btree_insert_int(slotset *ss, btreenode *node, 
    CmiInt8 startslot, CmiInt8 nslots)
{
  /*** binary search for the place to insert ***/

  auto helper = [&](const int index) -> insert_ret_val
  {
    if (node->child[index] != NULL) /* take child */
    {
      insert_ret_val irv = btree_insert_int(ss, node->child[index], startslot, nslots);
      if (irv.btn != NULL) /* merge return value */
      {
        /* insert */
        for (int i = node->num_blocks; i > index; i--)
        {
          copy_slotblock(node->blocks[i], node->blocks[i-1]);
          node->child[i+1] = node->child[i];
        }

        copy_slotblock(node->blocks[index], irv.sb);
        node->child[index+1] = irv.btn;
        node->num_blocks++;

        if (node->num_blocks == TREE_NODE_SIZE)
        {
          /* split node */
          btreenode *new_node = create_btree_node();

          for (int i = TREE_NODE_MID+1; i < TREE_NODE_SIZE; i++)
          {
            int j = i - (TREE_NODE_MID+1);
            copy_slotblock(new_node->blocks[j], node->blocks[i]);
          }

          for (int i = TREE_NODE_MID+1; i <= TREE_NODE_SIZE; i++)
            new_node->child[i-(TREE_NODE_MID+1)] = node->child[i];

          node->num_blocks     = TREE_NODE_MID;
          new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

          irv.sb  = node->blocks[TREE_NODE_MID];
          irv.btn = new_node;
        }
        else
        {
          irv.btn = NULL;
        }
      }

      return irv;
    }
    else /* insert */
    {
      insert_ret_val irv{};

      for (int i = node->num_blocks; i > index; i--)
        copy_slotblock(node->blocks[i], node->blocks[i-1]);

      insert_slotblock(node->blocks[index], ss, startslot, nslots);
      node->num_blocks++;

      if (node->num_blocks == TREE_NODE_SIZE)
      {
        /* split node */
        btreenode *new_node = create_btree_node();

        for (int i = TREE_NODE_MID+1; i < TREE_NODE_SIZE; i++)
        {
          int j = i - (TREE_NODE_MID+1);
          copy_slotblock(new_node->blocks[j], node->blocks[i]);
        }

        node->num_blocks     = TREE_NODE_MID;
        new_node->num_blocks = TREE_NODE_SIZE - TREE_NODE_MID - 1;

        irv.sb  = node->blocks[TREE_NODE_MID];
        irv.btn = new_node;
      }
      else
      {
        irv.btn = NULL;
      }

      return irv;
    }
  };

  /* initialize */
  int index = node->num_blocks >> 1;
  int inc   = (index >> 1) + (node->num_blocks & 0x1);

  /* loop until an insertion happens */
  while (1)
  {
    if (startslot < node->blocks[index].startslot) /* look to the left */
    {
      if (index == 0 || startslot > node->blocks[index-1].startslot)
      {
        return helper(index);
      }
      else /* search to the left */
      {
        index -= inc;
        if ((inc >>= 1) == 0)
          inc = 1;
      }
    }
    else /* look to the right */
    {
      if (index == node->num_blocks-1 || startslot < node->blocks[index+1].startslot)
      {
        return helper(index+1);
      }
      else /* search to the right */
      {
        index += inc;
        if ((inc >>= 1) == 0)
          inc = 1;
      }
    }
  }
}

static btreenode *btree_insert(slotset *ss, btreenode *node, 
    CmiInt8 startslot, CmiInt8 nslots)
{
  /* check the b-tree root: if it's empty, insert the element in the
     first position */
  if (node->num_blocks == 0)
  {
    node->num_blocks = 1;
    insert_slotblock(node->blocks[0], ss, startslot, nslots);
  }
  else
  {
    /* insert into the b-tree */
    insert_ret_val irv = btree_insert_int(ss, node, startslot, nslots);

    /* if something was returned, we need a new root */
    if (irv.btn != NULL)
    {
      btreenode *new_root  = create_btree_node();
      new_root->num_blocks = 1;
      copy_slotblock(new_root->blocks[0], irv.sb);
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
    CmiInt8 startslot, slotblock *sb)
{
  int index;

  /* If sb is not NULL, we're sending sb down the tree to a leaf to be
     swapped with the next larger startslot so it can be deleted from
     a leaf node (deletions from non-leaf nodes are not allowed
     here).  At this point, the next larger startslot will always be
     found by taking the leftmost child.  */
  if (sb != NULL)
  {
    if (node->child[0] != NULL)
    {
      btree_delete_int(ss, node->child[0], startslot, sb);
      index = 0;
    }
    else
    {
      /* we're now at a leaf node, so the slotblock can be deleted

         first, copy slotblock 0 to the block passed down (sb) and
         delete the list array node  */
      list_delete(ss, sb);
      copy_slotblock(*sb, node->blocks[0]);

      /* delete the slotblock */
      for (int i = 0; i < (node->num_blocks - 1); i++)
        copy_slotblock(node->blocks[i], node->blocks[i+1]);

      node->num_blocks--;

      return;
    }
  }
  else
  {
    /*** Binary search for the slotblock to delete ***/

    /* initialize */
    index = node->num_blocks >> 1;
    int inc = (index >> 1) + (node->num_blocks & 0x1);

    /* loop until the slotblock with startslot is found */
    while (1)
    {
      if (startslot == node->blocks[index].startslot) /* found it */
      {
        if (node->child[index+1] != NULL)
        {               /* not a leaf */
          btree_delete_int(ss, node->child[index+1], 
              startslot, &(node->blocks[index]));
          break;
        }
        else /* is a leaf */
        {
          /* delete the slotblock */
          list_delete(ss, &(node->blocks[index]));
          for (int i = index; i < (node->num_blocks - 1); i++)
            copy_slotblock(node->blocks[i], node->blocks[i+1]);

          node->num_blocks--;
          return;
        }
      }
      else
      {
        if (startslot < node->blocks[index].startslot) /* look left */
        {
          if (index == 0 || startslot > node->blocks[index-1].startslot)
          {
            /* take left child */
            btree_delete_int(ss, node->child[index], startslot, sb);
            break;
          }
          else /* search left */
          {
            index -= inc;
            if ((inc >>= 1) == 0)
              inc = 1;
          }
        }
        else /* look right */
        {
          if (index == node->num_blocks - 1 || startslot < node->blocks[index+1].startslot)
          {
            /* take right child */
            btree_delete_int(ss, node->child[index+1], startslot, sb);
            break;
          }
          else /* search right */
          {
            index += inc;
            if ((inc >>= 1) == 0)
              inc = 1;
          }
        }
      }
    }
  }

  /* At this point, the desired slotblock has been removed, and we're
     going back up the tree.  We must check for deficient nodes that
     require the rotating or combining of elements to maintain a
     balanced b-tree. */
  int def_child = -1;

  /* check if one of the child nodes is deficient  */
  if (node->child[index]->num_blocks < TREE_NODE_MID)
    def_child = index;
  else if (node->child[index+1]->num_blocks < TREE_NODE_MID)
    def_child = index+1;

  if (def_child >= 0)
  {
    btreenode * dc = node->child[def_child];

    /* if there is a left sibling and it has enough elements, rotate */
    /* to the right */
    if (def_child != 0 &&
        node->child[def_child-1] != NULL &&
        node->child[def_child-1]->num_blocks > TREE_NODE_MID)
    {
      btreenode * const dcleft = node->child[def_child-1];
      slotblock & leftblock = node->blocks[def_child-1];

      /* move all elements in deficient child to the right */
      for (int i = dc->num_blocks; i > 0; i--)
        copy_slotblock(dc->blocks[i], dc->blocks[i-1]);
      for (int i = dc->num_blocks + 1; i > 0; i--)
        dc->child[i] = dc->child[i-1];

      /* move parent element to the deficient child */
      copy_slotblock(dc->blocks[0], leftblock);
      dc->num_blocks++;

      /* move the right-most child of the parent's left child to the
         left-most child of the formerly deficient child  */
      int j = dcleft->num_blocks;
      dc->child[0] = dcleft->child[j];

      /* move largest element from left child up to the parent */
      j--;
      copy_slotblock(leftblock, dcleft->blocks[j]);
      dcleft->num_blocks--;

    }
    /* otherwise, if there is a right sibling and it has enough */
    /* elements, rotate to the left */
    else if ((def_child+1) <= node->num_blocks &&
             node->child[def_child+1] != NULL &&
             node->child[def_child+1]->num_blocks > TREE_NODE_MID)
    {
      btreenode * const dcright = node->child[def_child+1];
      slotblock & rightblock = node->blocks[def_child];

      /* move parent element to the deficient child */
      int j = dc->num_blocks;
      copy_slotblock(dc->blocks[j], rightblock);
      dc->num_blocks++;

      /* move the left-most child of the parent's right child to the
         right-most child of the formerly deficient child  */
      j++;
      dc->child[j] = dcright->child[0];

      /* move smallest element from right child up to the parent */
      copy_slotblock(rightblock, dcright->blocks[0]);
      dcright->num_blocks--;

      /* move all elements in the parent's right child to the left  */
      for (int i = 0; i < dcright->num_blocks; i++)
        copy_slotblock(dcright->blocks[i], dcright->blocks[i+1]);
      for (int i = 0; i < dcright->num_blocks + 1; i++)
        dcright->child[i] = dcright->child[i+1];
    }
    /* otherwise, merge the deficient node, parent, and the parent's
       other child (one of the deficient node's siblings) by dropping
       the parent down to the level of the children */
    else
    {
      btreenode * const left = node->child[index];
      btreenode *& right = node->child[index+1];

      /* move the parent element into the left child node */
      int j = left->num_blocks;
      copy_slotblock(left->blocks[j], node->blocks[index]);
      left->num_blocks++;

      /* move the elements and children of the right child node to the */
      /* left child node */
      int num_left  = left->num_blocks;
      int num_right = right->num_blocks;
      for (int left_pos = num_left, right_pos = 0; left_pos < num_left + num_right; left_pos++, right_pos++)
        copy_slotblock(left->blocks[left_pos], right->blocks[right_pos]);
      for (int left_pos = num_left, right_pos = 0; left_pos < num_left + num_right + 1; left_pos++, right_pos++)
        left->child[left_pos] = right->child[right_pos];

      left->num_blocks = num_left + num_right;

      /* delete the right child node */
      free_reentrant(right);
      right = NULL;

      /* update the parent node */
      node->num_blocks--;
      for (int i = index; i < node->num_blocks; i++)
      {
        copy_slotblock(node->blocks[i], node->blocks[i+1]);
        node->child[i+1] = node->child[i+2];
      }
    }
  }
}

static btreenode *btree_delete(slotset *ss, btreenode *node, CmiInt8 startslot)
{
  /* delete element from the b-tree */
  btree_delete_int(ss, node, startslot, NULL);

  /* if the root node is empty (from a combine operation on the tree),
     the left-most child of the root becomes the new root, unless the
     left-most child is NULL, in which case we leave the root node
     empty but not NULL */
  if (node->num_blocks == 0)
  {
    if (node->child[0] != NULL)
    {
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

static slotset *new_slotset(CmiInt8 startslot, CmiInt8 nslots)
{
  /* allocate memory for the slotset */
  auto ss = (slotset *)malloc_reentrant(sizeof(slotset));

  /* allocate memory for the b-tree */
  ss->btree_root = create_btree_node();

  /* initialize the b-tree */
  ss->btree_root->num_blocks = 1;
  slotblock & block = ss->btree_root->blocks[0];
  block.startslot = startslot;
  block.nslots    = nslots;

  /* initialize the list array */
  for (int i = 0; i < LIST_ARRAY_SIZE; i++)
    ss->list_array[i] = NULL;

  auto bin = (dllnode *)malloc_reentrant(sizeof(dllnode));
  bin->previous = NULL;
  bin->next = NULL;
  bin->sb = &block;

  int list_bin = find_list_bin(nslots);
  ss->list_array[list_bin] = bin;
  block.listblock = bin;

  return ss;
}

/*****************************************************************
 * Finds a slotblock containing at least nslots memory slots and
 * returns the startslot of that slotblock; returns -1 if no such
 * slotblock exists.
 *****************************************************************/

static CmiInt8 get_slots(slotset *ss, CmiInt8 nslots)
{
  /* calculate the smallest bin (list) to look in first */
  int start_list = find_list_bin(nslots);

  /* search for a slotblock with enough slots */
  for (int i = start_list; i < LIST_ARRAY_SIZE; i++)
  {
    dllnode *dlln = ss->list_array[i];
    while (dlln != NULL)
    {
      if (dlln->sb->nslots >= nslots)
        return dlln->sb->startslot;

      dlln = dlln->next;
    }
  }

  /* no such slotblock exists */
  return -1;
}

/*****************************************************************
 * Grab a slotblock with the specified range of blocks (nslots blocks
 * starting at sslot).  This is different from get_slots because
 * grab_slots specifies the slots to be grabbed and actually grabs
 * them (removes them from the set of free slots).  get_slots only
 * finds a set of free slots.
 *****************************************************************/

static void grab_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots)
{
  slotblock *sb = find_btree_slotblock(ss->btree_root, sslot);

  if (sb == NULL)
    CmiAbort("requested a non-existent slotblock\n");

  if (sb->startslot == sslot)
  {
    /* range is exact range of slotblock - delete block from tree */
    if (sb->nslots == nslots)
    {
      ss->btree_root = btree_delete(ss, ss->btree_root, sslot);
    }
    /* range is at beginning of slotblock - update block range */
    else
    {
      CmiInt8 old_nslots = sb->nslots;
      sb->startslot     += nslots;
      sb->nslots        -= nslots;
      list_move(ss, sb->listblock, old_nslots);
    }
  }
  else
  {
    /* range is at end of slotblock - update block range */
    CmiInt8 endslot = sb->startslot + sb->nslots - 1;
    if (endslot == (sslot + nslots - 1))
    {
      CmiInt8 old_nslots = sb->nslots;
      sb->nslots        -= nslots;
      list_move(ss, sb->listblock, old_nslots);
    }

    /* range is in middle of slotblock - update block range with the */
    /* new lower range and insert a block with the new upper range */
    else
    {
      CmiInt8 old_nslots = sb->nslots;
      sb->nslots         = sslot - sb->startslot;
      list_move(ss, sb->listblock, old_nslots);
      ss->btree_root = btree_insert(ss, ss->btree_root, sslot + nslots,
          endslot - (sslot + nslots) + 1);
    }
  }
}

/*****************************************************************
 * Frees nslots memory slots starting at sslot by either adding them
 * to one of the slotblocks that exists (if the slot ranges are
 * contiguous) or by creating a new slotblock
 *****************************************************************/

static void free_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots)
{
  slotblock *sb_low  = find_btree_slotblock(ss->btree_root, sslot - 1);
  slotblock *sb_high = find_btree_slotblock(ss->btree_root, sslot + nslots);

  if (sb_low == NULL)
  {
    if (sb_high == NULL)
    {
      /* there is no adjacent slotblock, so create a new one and */
      /* insert it in the b-tree */
      ss->btree_root = btree_insert(ss, ss->btree_root, sslot, nslots);
    }
    else
    {
      /* there is an adjacent slotblock to the right, so update its range */
      CmiInt8 old_nslots = sb_high->nslots;
      sb_high->startslot = sslot;
      sb_high->nslots   += nslots;
      list_move(ss, sb_high->listblock, old_nslots);
    }
  }
  else
  {
    if (sb_high == NULL)
    {
      /* there is an adjacent slotblock to the left, so update its range */
      CmiInt8 old_nslots  = sb_low->nslots;
      sb_low->nslots     += nslots;
      list_move(ss, sb_low->listblock, old_nslots);
    }
    else
    {
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

static void delete_btree(btreenode *node)
{
  for (int i = 0; i <= node->num_blocks; i++)
  {
    if (node->child[i] != NULL)
    {
      delete_btree(node->child[i]);
      free_reentrant(node->child[i]);
    }
    else
    {
      return;
    }
  }
}

/*****************************************************************
 * Free the allocated memory of the list array
 *****************************************************************/

static void delete_list_array(slotset *ss)
{
  for (int i = 0; i < LIST_ARRAY_SIZE; i++)
  {
    dllnode *dlln = ss->list_array[i];
    if (dlln != NULL)
    {
      while (dlln->next != NULL)
      {
        dlln = dlln->next;
      }
      while (dlln->previous != NULL)
      {
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

static void delete_slotset(slotset *ss)
{
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
static void print_btree_node(btreenode *node, int node_num)
{
  int i;
  CmiPrintf("Node %2d: ", node_num);
  for (i = 0; i < node->num_blocks; i++)
    CmiPrintf("%d:[%lld,%lld] ", i, node->blocks[i].startslot, node->blocks[i].nslots);
  CmiPrintf("\n");
}

/* returns 1 if there's another level to print; 0 if not */
static int print_btree_level(btreenode *node, int level, int current_level, int node_num)
{
  int another_level;
  for (int i = 0; i <= node->num_blocks; i++)
  {
    if (current_level == level)
    {
      print_btree_node(node, node_num);
      return node->child[0] != NULL;
    }
    else
    {
      another_level = print_btree_level(node->child[i], level, 
          current_level + 1, i);
    }
  }
  return another_level;
}

static void print_btree_top_down(btreenode *node)
{
  int level = 0;
  int another_level;
  do
  {
    CmiPrintf("B-tree Level %d\n", level);
    CmiPrintf("---------------\n");
    another_level = print_btree_level(node, level, 0, 0);
    level++;
    CmiPrintf("\n\n");
  } while (another_level);
}

/*****************************************************************
 * Print the contents of the list array on the screen
 *****************************************************************/

static void print_list_array(slotset *ss)
{
  CmiPrintf("List Array\n");
  CmiPrintf("----------\n");
  for (int i = 0; i < LIST_ARRAY_SIZE; i++)
  {
    CmiPrintf("List %2d: ", i);
    dllnode *dlln = ss->list_array[i];
    while (dlln != NULL)
    {
      if (dlln->previous != NULL)
        CmiPrintf("<->");
      else
        CmiPrintf(" ->");

      CmiPrintf("[%lld,%lld]", dlln->sb->startslot, dlln->sb->nslots);
      dlln = dlln->next;
    }
    CmiPrintf("\n");
  }
}

#if ISOMALLOC_DEBUG
static void print_slots(slotset *ss)
{
  print_btree_top_down(ss->btree_root);
  print_list_array(ss);
}
#else
#  define print_slots(ss) /*empty*/
#endif

/*This version of the allocate/deallocate calls are used if the 
  real mmap versions are disabled.*/
static int disabled_map_warned=0;
static void *disabled_map(int nBytes) 
{
  if (!disabled_map_warned) {
    disabled_map_warned=1;
    if (CmiMyPe()==0)
      CmiError("Charm++> Warning: Isomalloc is uninitialized."
          " You won't be able to migrate threads.\n");
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
    CmiPrintf("Charm++> Disabling isomalloc because %s.\n", why);
}

#if ! CMK_HAS_MMAP
/****************** Manipulate memory map (Win32 non-version) *****************/
static void *call_mmap_fixed(void *addr,size_t len) {
  CmiAbort("isomalloc.C: mmap_fixed should never be called here.");
  return NULL;
}
static void *call_mmap_anywhere(size_t len) {
  CmiAbort("isomalloc.C: mmap_anywhere should never be called here.");
  return NULL;
}
static void call_munmap(void *addr,size_t len) {
  CmiAbort("isomalloc.C: munmap should never be called here.");
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
    DEBUG_PRINT("[%d] tried to mmap %p, but encountered error\n",CmiMyPe(),addr);
    return NULL;
  }
  if (pa != addr)
  { /*Map worked, but gave us back the wrong place*/
    DEBUG_PRINT("[%d] tried to mmap %p, but got %p back\n",CmiMyPe(),addr,pa);
    call_munmap(addr,slotsize*nslots);
    return NULL;
  }
  DEBUG_PRINT("[%d] mmap'd slots %ld-%ld to address %p\n",CmiMyPe(),slot,slot+nslots-1,addr);
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
  DEBUG_PRINT("[%d] munmap'd slots %ld-%ld from address %p\n",CmiMyPe(),slot,slot+nslots-1,addr);
}

static void map_failed(CmiInt8 s,CmiInt8 n)
{
  void *addr=slot2addr(s);
  CmiError("Charm++> Isomalloc map failed to allocate %d bytes at %p, errno: %d.\n",
      slotsize*n, addr, errno);
  CmiAbort("Exiting\n");
}



/************ Address space voodoo: find free address range **********/

CpvStaticDeclare(slotset *, myss); /*My managed slots*/

#if CMK_USE_MEMPOOL_ISOMALLOC && !USE_ISOMEMPOOL
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
    DEBUG_PRINT("[%d] Skipping unmappable space at %p\n",CmiMyPe(),loc);
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
  DEBUG_PRINT("good_range: %lld, %d\n", quant, n);
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
  DEBUG_PRINT("[%d] Checking at %p - %p\n",CmiMyPe(),start,end);

  /* Check the middle of the range */
  if (!good_range(start,len,256)) {
    /* Try to split into subranges: */
    int i,n=2;
    DEBUG_PRINT("[%d] Trying to split bad address space at %p - %p...\n",CmiMyPe(),start,end);
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
    DEBUG_PRINT("[%d] Address space at %p - %p is largest\n",CmiMyPe(),start,end);

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
    char *holeEnd=(char *)(intptr_t)-1;

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
    DEBUG_PRINT("[%d] Memory map: %p - %p (len: %lu => %lu) %s \n",CmiMyPe(),
                regions[i].start,regions[i].start+regions[i].len,
                old.len, regions[i].len, regions[i].type);
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
#if CMK_HAS_USLEEP
      if (retry++ < 5) { usleep(rand()%10000); continue; }
      else retry = 0;
#endif
      size=(double)size/shrink; /* shrink request */
      if (size<=0) return 0; /* mmap doesn't work */
    }
    else { /* this allocation size is available */
      DEBUG_PRINT("[%d] available: %p, %lld\n", CmiMyPe(), range, size);
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
  destRegion->start = (char *)good_range;
  destRegion->len=good_size;
#if ISOMALLOC_DEBUG
  pid_t pid = getpid();
  {
    char s[128];
    sprintf(s, "cat /proc/%d/maps", pid);
    system(s);
  }
  DEBUG_PRINT("[%d] try_largest_mmap_region: %p, %lld\n", CmiMyPe(), good_range, good_size);
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
#if CMK_USE_MEMPOOL_ISOMALLOC || USE_ISOMEMPOOL
    slotsize=1024*1024;
#else
    slotsize=16*1024;
#endif 
    pagesize = CmiGetPageSize();
    slotsize=(slotsize+pagesize-1) & ~(pagesize-1);

#if ISOMALLOC_DEBUG
    if (CmiMyPe() == 0)
      DEBUG_PRINT("[%d] Using slotsize of %d\n", CmiMyPe(), slotsize);
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
      DEBUG_PRINT("[%d] Isomalloc memory region: %p - %p (%d megs)\n",CmiMyPe(),
                  freeRegion.start,freeRegion.start+freeRegion.len,freeRegion.len/meg);
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
            if (read(fd, &ss, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
              CmiAbort("Isomalloc> call to read() failed during restart!");
            }
            if (read(fd, &ee, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
              CmiAbort("Isomalloc> call to read() failed during restart!");
            }
            close(fd);
            if (ss < s || ee > e)
                CmiAbort("isomalloc_sync failed during restart, virtual memory regions do not overlap.");
            else {
                freeRegion.start = (char *)ss;
                freeRegion.len = (char *)ee -(char *)ss;
            }
            CmiPrintf("Charm++> Consolidated Isomalloc memory region at restart: %p - %p (%d MB).\n",freeRegion.start,freeRegion.start+freeRegion.len,freeRegion.len/meg);
            goto AFTER_SYNC;
        }
#endif
    if (CmiMyRank() == 0 && freeRegion.len > 0u) {
      if (CmiBarrier() == -1 && CmiMyPe()==0) 
        CmiAbort("Charm++ Error> +isomalloc_sync requires CmiBarrier() implemented.\n");
      else {
	/* cppcheck-suppress uninitStructMember */
        uintptr_t s = (uintptr_t)freeRegion.start;
	/* cppcheck-suppress uninitStructMember */
        uintptr_t e = (uintptr_t)freeRegion.start+freeRegion.len;
        int fd, i;
        char fname[128];

        if (CmiMyNode() == 0)
          CmiPrintf("Charm++> Synchronizing isomalloc memory region...\n");

        sprintf(fname,".isomalloc.%d", CmiMyNode());

        /* remove file before writing for safe */
        unlink(fname);
#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        if (system("sync") == -1) {
          CmiAbort("Isomalloc> call to system(\"sync\") failed while synchronizing memory regions!");
        }
#endif

        CmiBarrier();

        /* write region into file */
        while ((fd = open(fname, O_WRONLY|O_TRUNC|O_CREAT, 0644)) == -1) 
#ifndef __MINGW_H
          CMK_CPV_IS_SMP
#endif
            ;
        if (write(fd, &s, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
          CmiAbort("Isomalloc> call to write() failed while synchronizing memory regions!");
        }
        if (write(fd, &e, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
          CmiAbort("Isomalloc> call to write() failed while synchronizing memory regions!");
        }
        close(fd);

#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        if (system("sync") == -1) {
          CmiAbort("Isomalloc> call to system(\"sync\") failed while synchronizing memory regions!");
        }
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
          if (read(fd, &ss, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
            CmiAbort("Isomalloc> call to read() failed while synchronizing memory regions!");
          }
          if (read(fd, &ee, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
            CmiAbort("Isomalloc> call to read() failed while synchronizing memory regions!");
          }
#if ISOMALLOC_DEBUG
          if (CmiMyPe() == 0)
            DEBUG_PRINT("[%d] load node %d isomalloc region: %lx %lx.\n", CmiMyPe(), i, ss, ee);
#endif
          close(fd);
          if (ss>s) s = ss;
          if (ee<e) e = ee;
        }

        CmiBarrier();

        unlink(fname);
#if CMK_HAS_SYNC && ! CMK_DISABLE_SYNC
        if (system("sync") == -1) {
          CmiAbort("Isomalloc> call to system(\"sync\") failed while synchronizing memory regions!");
        }
#endif

        /* update */
        if (s > e)  {
          if (CmiMyPe()==0) CmiPrintf("[%d] Invalid isomalloc region: %lx - %lx.\n", CmiMyPe(), s, e);
          CmiAbort("Isomalloc> failed to find consolidated isomalloc region!");
        }
        freeRegion.start = (char *)(uintptr_t)s;
        freeRegion.len = (char *)(uintptr_t)e -(char *)(uintptr_t)s;

        if (CmiMyPe() == 0)
          CmiPrintf("Charm++> Consolidated Isomalloc memory region: %p - %p (%d MB).\n",
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
                    if (write(fd, &s, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
                      CmiAbort("Isomalloc> call to write() failed while synchronizing memory regions!");
                    }
                    if (write(fd, &e, sizeof(CmiUInt8)) != sizeof(CmiUInt8)) {
                      CmiAbort("Isomalloc> call to write() failed while synchronizing memory regions!");
                    }
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

    DEBUG_PRINT("[%d] Can isomalloc up to %lu MB per PE\n",CmiMyPe(),
                ((memRange_t)numslots)*slotsize/meg);
  }

  /*SMP Mode: wait here for rank 0 to initialize numslots before calculating myss*/
  CmiNodeAllBarrier(); 

  CpvInitialize(slotset *, myss);
  CpvAccess(myss) = NULL;

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
  CmiAssignOnce(&grab_remote_idx, CmiRegisterHandler((CmiHandler)grab_remote));
  CmiAssignOnce(&free_remote_idx, CmiRegisterHandler((CmiHandler)free_remote));
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

#if USE_ISOMEMPOOL

/************** Isomempool ***************/

#if ISOMEMPOOL_DEBUG
# define VERIFY_RBTREE
#endif

struct isomempool
{
  /*
   * The main purpose of this class is batching, to avoid wasting mmapped space.
   *
   * Definitions:
   * Blocks are the units of storage underlying the pool.
   * Regions are divisions within a block, either allocated space or empty space between
   * allocations.
   *
   * The pool prefixes each allocation with a small header forming a doubly linked list,
   * to allow flexible insertion, removal, and examination of neighbors.
   *
   * Each block has a header containing its "num_slots", which corresponds to its size. In
   * combination with the pointer to the block's beginning, this is all the information
   * necessary to unmap, migrate, and remap. Block headers also form a linked list of all
   * blocks in the pool.
   *
   * Calls to alloc() are accelerated by a cache structure keyed with relevant information:
   * a red-black tree of free space segments keyed by their sizes.
   *
   * Each Isomalloc context has its own mempool.
   *
   * Potential future improvements:
   * - Store RBTree::Nodes in their own blocks to improve cache locality of lookup
   * - Examine if __restrict is helpful anywhere
   * - Use blocks and the Isomalloc context instead of malloc_reentrant in slot btree
   * - Merge adjacent blocks, if worth the bookkeeping
   * - Simplify the code. Less time bookkeeping = more time in the user's code.
   */

  using BitCarrier = unsigned char;

  struct RegionHeader
  {
    // tag structs
    struct Occupied
    {
      explicit Occupied(bool v = false) : value{v} { }
      explicit Occupied(uintptr_t v) : value{(bool)v} { }
      operator bool() const { return value; }
    private:
      bool value;
    };
    struct Last
    {
      explicit Last(bool v = false) : value{v} { }
      explicit Last(uintptr_t v) : value{(bool)v} { }
      operator bool() const { return value; }
    private:
      bool value;
    };

    RegionHeader(RegionHeader *p, RegionHeader *n, Occupied o, Last l)
      : prevfield{p}
      , nextfield{n}
      , occupiedfield{o}
      , lastfield{l}
    {
      IMP_DBG("[%p] RegionHeader::RegionHeader(%p, %p, %u, %u)\n",
              this, p, n, (int)(bool)o, (int)(bool)l);
    }
    RegionHeader(RegionHeader *p, RegionHeader *n, Occupied o, BitCarrier l)
      : prevfield{p}
      , nextfield{n}
      , occupiedfield{o}
      , lastfield{l}
    {
      IMP_DBG("[%p] RegionHeader::RegionHeader(%p, %p, %u, %u)\n",
              this, p, n, (int)(bool)o, (int)(bool)l);
    }

    RegionHeader * prev() const
    {
      return prevfield;
    }
    RegionHeader * next() const
    {
      return nextfield;
    }

    void setPrev(RegionHeader *ptr)
    {
      IMP_DBG("[%p] RegionHeader::setPrev(%p)\n", this, ptr);

      prevfield = ptr;
    }
    void setNext(RegionHeader *ptr)
    {
      IMP_DBG("[%p] RegionHeader::setNext(%p)\n", this, ptr);

      nextfield = ptr;
    }

    BitCarrier occupied() const
    {
      return occupiedfield;
    }
    bool isOccupied() const
    {
      return occupiedfield;
    }
    void setOccupied(bool o)
    {
      IMP_DBG("[%p] RegionHeader::setOccupied(%u)\n", this, (unsigned int)o);

      occupiedfield = (BitCarrier)o;
    }
    void setOccupied(BitCarrier o)
    {
      IMP_DBG("[%p] RegionHeader::setOccupied(%u)\n", this, (unsigned int)o);

      occupiedfield = o;
    }

    BitCarrier last() const
    {
      return lastfield;
    }
    void setLast(bool l)
    {
      IMP_DBG("[%p] RegionHeader::setLast(%u)\n", this, (unsigned int)l);

      lastfield = (BitCarrier)l;
    }
    void setLast(BitCarrier l)
    {
      IMP_DBG("[%p] RegionHeader::setLast(%u)\n", this, (unsigned int)l);

      lastfield = l;
    }

    bool isFirst() const
    {
      return ((const BlockHeader *)prev())->firstRegion() == this;
    }
    bool isLast() const
    {
      return lastfield;
    }

    bool prevIsEmpty() const
    {
      return !isFirst() && !prev()->isOccupied();
    }
    bool nextIsEmpty() const
    {
      return !isLast() && !next()->isOccupied();
    }

    size_t size() const
    {
      return (const uint8_t *)next() - (const uint8_t *)this;
    }

  private:
    RegionHeader * prevfield;
    RegionHeader * nextfield;
    unsigned char occupiedfield;
    unsigned char lastfield;
  };

  using Occupied = RegionHeader::Occupied;
  using Last = RegionHeader::Last;

  struct BlockHeader
  {
    void setFirstRegion(RegionHeader *p)
    {
      IMP_DBG("[%p] BlockHeader::setFirstRegion(%p)\n", this, p);

      first_region = p;
    }

    RegionHeader * firstRegion() const
    {
      return first_region;
    }

  private:
    // alignment_filler requires some kind of measurement instead of adding sizeof(BlockHeader)
    RegionHeader *first_region;
  public:
    BlockHeader *prev, *next;
    CmiInt8 num_slots;
  };

  struct IsomallocBlockRecord
  {
    IsomallocBlockRecord() = delete;
    IsomallocBlockRecord(const BlockHeader *block_header)
      : slot{addr2slot(block_header)}
      , num_slots{block_header->num_slots}
    {
    }
    IsomallocBlockRecord(uintptr_t block_location, CmiInt8 n)
      : slot{addr2slot((const void *)block_location)}
      , num_slots{n}
    {
    }
    IsomallocBlockRecord(size_t requested_size)
    {
      const CmiInt8 n = length2slots(requested_size);

      /* Always satisfy mallocs with local slots: */
      const CmiInt8 s = get_slots(CpvAccess(myss), n);
      if (s == -1)
      {
        CmiError("Not enough address space left on processor %d to isomalloc %d bytes!\n",
                  CmiMyPe(), requested_size);
        CmiAbort("Out of virtual address space for isomalloc");
      }

      slot = s;
      num_slots = n;
    }

    CmiInt8 numSlots() const
    {
      return num_slots;
    }
    uint8_t *ptr() const
    {
      return (uint8_t *)slot2addr(slot);
    }
    size_t size() const
    {
      return numSlots() * slotsize;
    }
    uint8_t *endpoint() const
    {
      return ptr() + size();
    }

    void *map() const
    {
      grab_slots(CpvAccess(myss), slot, numSlots());
      return remap();
    }
    void *remap() const
    {
      void *myptr;
      for (int i = 0; i < 5; i++)
      {
        myptr = map_slots(slot, numSlots());
        if (myptr != nullptr)
          break;

#if CMK_HAS_USLEEP
        const auto err = errno;
        if (err == ENOMEM)
        {
          usleep(rand() % 1000);
          continue;
        }
        else
          break;
#endif
      }
      if (myptr == nullptr)
        map_failed(slot, numSlots());

      IMP_DBG("Isomalloc block %p-%p mapped (%zu)\n", ptr(), ptr() + size(), size());

      return (uint8_t *)myptr;
    }
    void unmap() const
    {
      call_munmap(ptr(), size());
      IMP_DBG("Isomalloc block %p-%p unmapped (%zu)\n", ptr(), ptr() + size(), size());
    }

  private:
    CmiInt8 slot;
    CmiInt8 num_slots;
  };

  struct RBTree
  {
    enum color : uintptr_t
    {
      RED = 0,
      BLACK = 1
    };

    struct Node
    {
      // tag structs
      struct Color
      {
        explicit Color(uintptr_t v) : value{v} { }
        Color(RBTree::color v = RED) : value{(uintptr_t)v} { }
        operator uintptr_t() const { return value; }
        operator RBTree::color() const { return (RBTree::color)value; }
      private:
        uintptr_t value;
      };

      size_t key() const
      {
        return size();
      }
      size_t size() const
      {
        return header()->size();
      }
      uint8_t * ptr() const
      {
        return (uint8_t *)location();
      }
      RegionHeader * header() const
      {
        return (RegionHeader *)location();
      }
      uintptr_t location() const
      {
        return (uintptr_t)this - sizeof(RegionHeader);
      }

      // tree node data
      Node * leftField{};
      Node * rightField{};
      Node * parentField{};
      Color colorField{};

      bool isRoot() const
      {
        return parentField == nullptr;
      }
      Node * parent()
      {
        return parentField;
      }
      Node * left()
      {
        return leftField;
      }
      Node * right()
      {
        return rightField;
      }
      RBTree::color color() const
      {
        return colorField;
      }
      void setColor(RBTree::color c)
      {
        colorField = c;
      }

      Node * grandparent()
      {
        CmiAssert(!this->isRoot());           /* Not the root node */
        CmiAssert(!this->parent()->isRoot()); /* Not child of root */
        return this->parent()->parent();
      }
      Node * sibling()
      {
        CmiAssert(!this->isRoot());           /* Root node has no sibling */
        if (this == this->parent()->left())
          return this->parent()->right();
        else
          return this->parent()->left();
      }
      Node * parent_sibling()
      {
        CmiAssert(!this->isRoot());           /* Root node has no parent-sibling */
        CmiAssert(!this->parent()->isRoot()); /* Children of root have no parent-sibling */
        return this->parent()->sibling();
      }
      Node * maximum_node()
      {
        Node * n = this;
        Node * right;
        while ((right = n->right()) != nullptr)
        {
          n = right;
        }
        return n;
      }
    };

    Node * lookup_region(size_t, size_t, size_t);
    void insert(Node *);
    void remove(Node *);

    void pup(PUP::er & p)
    {
      if (p.isUnpacking())
      {
        uintptr_t r;
        p | r;
        root = (Node *)r;
      }
      else
      {
        uintptr_t r = (uintptr_t)root;
        p | r;
      }
    }

  private:

    Node * root{};

    static color node_color(const RBTree::Node * n)
    {
      return n == nullptr ? BLACK : n->color();
    }

    void replace_node(Node * oldn, Node * newn);
    void rotate_left(Node * n);
    void rotate_right(Node * n);

    void insert_case1(Node * n);
    void insert_case2(Node * n);
    void insert_case3(Node * n);
    void insert_case4(Node * n);
    void insert_case5(Node * n);

    void delete_case1(Node * n);
    void delete_case2(Node * n);
    void delete_case3(Node * n);
    void delete_case4(Node * n);
    void delete_case5(Node * n);
    void delete_case6(Node * n);

#ifdef VERIFY_RBTREE
    void verify_properties();
    static void verify_property_1(Node * n);
    static void verify_property_2(Node * root);
    static void verify_property_4(Node * n);
    static void verify_property_5_helper(Node * n, int black_count, int * path_black_count);
    static void verify_property_5(Node * root);
#endif
  };

  using EmptyRegion = RBTree::Node;

  static constexpr size_t minimum_alignment = ALIGN_BYTES;

  // roughly the threshold where bookkeeping for a free region takes more space than the region itself
  static constexpr size_t minimum_empty_region_size = sizeof(RegionHeader) + sizeof(EmptyRegion);
  static_assert(minimum_empty_region_size > 0, "region sizes cannot be zero");
  static_assert(minimum_empty_region_size >= sizeof(RegionHeader), "regions must allow space for a header");

  isomempool()
  {
    IMP_DBG("[%p] isomempool::isomempool()\n", this);
  }

  ~isomempool()
  {
    IMP_DBG("[%p] isomempool::~isomempool()\n", this);

    clearBlocks();
  }

  void print_contents() const
  {
    CmiPrintf("[%p] isomempool::print_contents()\n", this);

    size_t count = num_blocks;
    CmiPrintf("Blocks: %zu\n", count);

    BlockHeader *block_header = first_block;
    for (size_t i = 0; i < count; ++i)
    {
      const uintptr_t address = (uintptr_t)block_header;
      const size_t size = block_header->num_slots * slotsize;
      CmiPrintf("  0x%016zx-0x%016zx %zu\n", address, address + size, size);

      const RegionHeader *node = block_header->firstRegion();
      bool isLast;
      do
      {
        const RegionHeader *next = node->next();

        const size_t size = (const uint8_t *)next - (const uint8_t *)node;
        CmiPrintf("    0x%016zx-0x%016zx %zu\n", node, next, size);

        isLast = node->isLast();
        node = next;
      }
      while (!isLast);

      block_header = block_header->next;
    }
  }

  static size_t get_alignment_filler(const void *ptr, size_t alignment)
  {
    CmiAssert((alignment & (alignment-1)) == 0); // assuming alignment is a power of two or zero
    return (alignment - ((uintptr_t)ptr & (alignment-1))) & (alignment-1);
    // return (alignment - ptr % alignment) % alignment; // non-power of two version
  }

  static size_t minimum_alignment_filler(const void *ptr)
  {
    return get_alignment_filler(ptr, minimum_alignment);
  }

  void *alloc(size_t size, const size_t alignment = minimum_alignment, size_t alignment_offset = 0);
  void free(void *user_ptr);

  void pup(PUP::er & p);

private:

  // canonical data
  BlockHeader *first_block{};
  size_t num_blocks{};
  RBTree empty_tree;


  void *allocFromEmptyRegion(const size_t size, EmptyRegion * target, const size_t alignment_filler)
  {
    const size_t region_size = target->size();
    RegionHeader * const empty_header = target->header();
    empty_tree.remove(target);

    IMP_DBG("[%p] isomempool::alloc(%zu, ..., ...) adding to block by erasing and filling region %p of size %zu\n",
            this, size, empty_header, region_size);

    CmiAssert(!empty_header->prevIsEmpty());
    CmiAssert(!empty_header->nextIsEmpty());

    // Extract information from the empty region's header before it is (potentially) overwritten.
    const bool empty_header_first = empty_header->isFirst();
    RegionHeader * const empty_header_prev = empty_header->prev();
    RegionHeader * const empty_header_next = empty_header->next();
    const auto empty_header_last = empty_header->last();

    const auto header_ptr = (RegionHeader *)((uint8_t *)empty_header + alignment_filler);

    RegionHeader * header_prev;
    RegionHeader * header_next;
    BitCarrier header_last;

    if (alignment_filler >= minimum_empty_region_size + minimum_alignment_filler(empty_header))
    {
      // Record a new empty region preceding the new occupied region.
      IMP_DBG("[%p] isomempool::alloc(%zu, ..., ...) inserting empty region {%zu, %p} left pad existing region\n",
              this, size, alignment_filler, empty_header);

      header_prev = new (empty_header) RegionHeader{empty_header_prev, header_ptr,
                                                    Occupied{false}, Last{false}};

      auto empty_region = new (header_prev + 1) EmptyRegion{};
      empty_tree.insert(empty_region);

      if (empty_header_first)
      {
        const auto block_header = (BlockHeader *)empty_header_prev;
        block_header->setFirstRegion(header_prev);
      }
      else
      {
        empty_header_prev->setNext(header_prev);
        empty_header_prev->setLast(false);
      }

      CmiAssert(!header_prev->prevIsEmpty());
    }
    else
    {
      // Directly link the new occupied region to the erased empty region's left neighbor.
      header_prev = empty_header_prev;

      if (empty_header_first)
      {
        const auto block_header = (BlockHeader *)empty_header_prev;
        block_header->setFirstRegion(header_ptr);
      }
    }

    const size_t size_difference = region_size - size - alignment_filler;
    auto follow_ptr = (uint8_t *)header_ptr + size;
    const size_t follow_alignment_filler = minimum_alignment_filler(follow_ptr);
    follow_ptr += follow_alignment_filler;

    if (size_difference >= minimum_empty_region_size + follow_alignment_filler)
    {
      // Record a new empty region following the new occupied region.
      IMP_DBG("[%p] isomempool::alloc(%zu, ..., ...) inserting empty region {%zu, %p} right pad existing region\n",
              this, size, size_difference, follow_ptr);

      header_next = new (follow_ptr) RegionHeader{header_ptr,
                                                  empty_header_next, Occupied{false}, empty_header_last};

      auto empty_region = new (header_next + 1) EmptyRegion{};
      empty_tree.insert(empty_region);

      if (!header_next->isLast())
        empty_header_next->setPrev(header_next);

      CmiAssert(!header_next->nextIsEmpty());

      header_last = Last{false};
    }
    else
    {
      // Directly link the new occupied region to the erased empty region's right neighbor.
      header_next = empty_header_next;
      header_last = empty_header_last;
    }

    // Instantiate the new region and connect its neighbors to it.
    const auto header = new (header_ptr) RegionHeader{header_prev, header_next, Occupied{true}, header_last};

    if (!header->isFirst())
    {
      header_prev->setNext(header);
      header_prev->setLast(false);
    }
    if (!header->isLast())
      header_next->setPrev(header);

    return header + 1;
  }

  void *allocFromNewBlock(const size_t size, const size_t alignment, const size_t alignment_offset)
  {
    const IsomallocBlockRecord block{sizeof(BlockHeader) + size + alignment};
    const size_t block_size = block.size();

    IMP_DBG("[%p] isomempool::alloc(%zu, %zu, %zu) creating block %p of size %zu\n",
            this, size, alignment, alignment_offset, block.ptr(), block_size);

    // Construct a new block.
    const auto block_header = new (block.map()) BlockHeader{};
    const auto block_header_sentinel = (RegionHeader *)block_header;

    block_header->num_slots = block.numSlots();
    block_header->prev = nullptr;
    block_header->next = first_block;
    if (first_block != nullptr)
      first_block->prev = block_header;
    first_block = block_header;
    ++num_blocks;

    const auto block_data_ptr = (uint8_t *)(block_header + 1);
    const size_t alignment_filler = get_alignment_filler(block_data_ptr + alignment_offset, alignment);
    const auto header_ptr = (RegionHeader *)(block_data_ptr + alignment_filler);

    RegionHeader * header_prev;
    RegionHeader * header_next;
    BitCarrier header_last;

    if (alignment_filler >= minimum_empty_region_size + minimum_alignment_filler(block_data_ptr))
    {
      // Record a new empty region preceding the new occupied region.
      IMP_DBG("[%p] isomempool::alloc(%zu, %zu, %zu) inserting empty region {%zu, %p} (left pad new block)\n",
              this, size, alignment, alignment_offset, alignment_filler, block_data_ptr);

      header_prev = new (block_data_ptr) RegionHeader{block_header_sentinel, header_ptr,
                                                      Occupied{false}, Last{false}};

      auto empty_region = new (header_prev + 1) EmptyRegion{};
      empty_tree.insert(empty_region);

      block_header->setFirstRegion(header_prev);
    }
    else
    {
      // Mark the new occupied region as the first in the block.
      header_prev = block_header_sentinel;

      block_header->setFirstRegion(header_ptr);
    }

    const size_t size_difference = block_size - sizeof(BlockHeader) - size - alignment_filler;
    auto follow_ptr = (uint8_t *)header_ptr + size;
    const size_t follow_alignment_filler = minimum_alignment_filler(follow_ptr);
    follow_ptr += follow_alignment_filler;

    if (size_difference >= minimum_empty_region_size + follow_alignment_filler)
    {
      // Record a new empty region following the new occupied region.
      IMP_DBG("[%p] isomempool::alloc(%zu, %zu, %zu) inserting empty region {%zu, %p} (right pad new block)\n",
              this, size, alignment, alignment_offset, size_difference, follow_ptr);

      header_next = new (follow_ptr) RegionHeader{header_ptr,
                                                  (RegionHeader *)block.endpoint(),
                                                  Occupied{false}, Last{true}};

      auto empty_region = new (header_next + 1) EmptyRegion{};
      empty_tree.insert(empty_region);

      header_last = Last{false};
    }
    else
    {
      // Mark the new occupied region as the last in the block.
      header_next = (RegionHeader *)block.endpoint();
      header_last = Last{true};
    }

    const auto header = new (header_ptr) RegionHeader{header_prev, header_next, Occupied{true}, header_last};

    return header + 1;
  }

  void clearBlocks()
  {
    size_t count = num_blocks;

    BlockHeader *block_header = first_block;
    for (size_t i = 0; i < count; ++i)
    {
      const IsomallocBlockRecord block{block_header};
      block_header = block_header->next;
      block.unmap();
    }

    first_block = nullptr;
    num_blocks = 0;
  }
};

void *isomempool::alloc(size_t size, const size_t alignment, size_t alignment_offset)
{
  IMP_DBG("[%p] isomempool::alloc(%zu, %zu, %zu)...\n", this, size, alignment, alignment_offset);

  size += sizeof(RegionHeader);
  alignment_offset += sizeof(RegionHeader);

  // Find the smallest empty region that can fit the data.
  EmptyRegion * node = empty_tree.lookup_region(size, alignment, alignment_offset);

  // Perform the allocation.
  void *ret = node != nullptr
              ? allocFromEmptyRegion(size, node, get_alignment_filler(node->ptr() + alignment_offset, alignment))
              : allocFromNewBlock(size, alignment, alignment_offset);

  IMP_DBG("[%p] isomempool::alloc(%zu, %zu, %zu) -> %p\n", this, size, alignment, alignment_offset, ret);

  return ret;
}

void isomempool::free(void *user_ptr)
{
  const auto orig_header = (RegionHeader *)user_ptr - 1;
  auto header = orig_header;

  IMP_DBG("[%p] isomempool::free(%p)... {header = %p}\n", this, user_ptr, orig_header);

  if (orig_header->prevIsEmpty())
  {
    // Merge with the free region to the left.
    RegionHeader * const header_prev = orig_header->prev();
    RegionHeader * const header_next = orig_header->next();
    const auto header_last = orig_header->last();

    auto empty_region = (EmptyRegion *)(header_prev + 1);
    IMP_DBG("[%p] isomempool::free(%p) erasing empty region {%zu, %p} (left)\n",
            this, user_ptr, empty_region->size(), header_prev);
    empty_tree.remove(empty_region);

    header = header_prev;
    header_prev->setNext(header_next);
    header_prev->setLast(header_last);
    if (!header_last)
      header_next->setPrev(header_prev);
  }

  if (orig_header->nextIsEmpty())
  {
    // Merge with the free region to the right.
    RegionHeader * const header_next = header->next();
    RegionHeader * const header_next_next = header_next->next();
    const auto header_next_last = header_next->last();

    const auto empty_region = (EmptyRegion *)(header_next + 1);
    IMP_DBG("[%p] isomempool::free(%p) erasing empty region {%zu, %p} (right)\n",
            this, user_ptr, empty_region->size(), header_next);
    empty_tree.remove(empty_region);

    header->setNext(header_next_next);
    header->setLast(header_next_last);
    if (!header_next_last)
      header_next_next->setPrev(header);
  }

  if (header->isFirst() && header->isLast()) // Is it the only remaining region in the block?
  {
    // Free the underlying block.
    const auto block_header = (BlockHeader *)header->prev();

    if (block_header->prev != nullptr)
      block_header->prev->next = block_header->next;
    else
      first_block = block_header->next;

    if (block_header->next != nullptr)
      block_header->next->prev = block_header->prev;

    --num_blocks;

    const IsomallocBlockRecord block{block_header};
    block.unmap();
  }
  else
  {
    // Record the new empty region.
    CmiAssert(!header->prevIsEmpty() && !header->nextIsEmpty());

    header->setOccupied(false);

    auto empty_region = new (header + 1) EmptyRegion{};
    empty_tree.insert(empty_region);

    IMP_DBG("[%p] isomempool::free(%p) inserted empty region {%zu, %p}\n",
            this, user_ptr, empty_region->size(), header);
  }
}

void isomempool::pup(PUP::er & p)
{
  IMP_DBG("[%p] isomempool::pup()%s%s%s%s\n", this, p.isSizing() ? " sizing" : "",
          p.isPacking() ? " packing" : "", p.isUnpacking() ? " unpacking" : "", p.isDeleting() ? " deleting" : "");

  p | empty_tree;

  if (p.isUnpacking())
  {
    size_t count;
    p | count;
    num_blocks = count;

    auto unpack = [this, &p]() -> BlockHeader *
    {
      uintptr_t address;
      CmiInt8 num_slots;

      p | address;
      p | num_slots;

      const IsomallocBlockRecord block{address, num_slots};
      block.remap();

      p((char *)address, num_slots * slotsize);

      const auto block_header = (BlockHeader *)address;
      // no need to reconnect the linked list thanks to isomalloc

      return block_header;
    };

    if (count > 0)
    {
      // unpack the first block separately so we can retrieve its address
      first_block = unpack();
      for (size_t i = 1; i < count; ++i)
        unpack();
    }
    else
    {
      first_block = nullptr;
    }
  }
  else
  {
    size_t count = num_blocks;
    p | count;

    BlockHeader *block_header = first_block;
    for (size_t i = 0; i < count; ++i)
    {
      uintptr_t address = (uintptr_t)block_header;
      CmiInt8 num_slots = block_header->num_slots;

      p | address;
      p | num_slots;

      p((char *)address, num_slots * slotsize);

      block_header = block_header->next;
    }
  }

  if (p.isDeleting())
  {
    clearBlocks();
  }
}

/*
Red-black tree code sourced from:
https://web.archive.org/web/20151002014345/http://en.literateprograms.org:80/Special:Downloadcode/Red-black_tree_(C)

Its authors have released all rights to it and placed it
in the public domain under the Creative Commons CC0 1.0 waiver.
https://creativecommons.org/publicdomain/zero/1.0/
*/

auto isomempool::RBTree::lookup_region(size_t size, size_t alignment, size_t alignment_offset) -> Node *
{
  Node * n = this->root;
  Node * result = nullptr;
  while (n != nullptr)
  {
    signed long long comp_result = get_alignment_filler(n->ptr() + alignment_offset, alignment) + size - n->size();
    if (comp_result == 0)
    {
      return n;
    }
    else if (comp_result < 0)
    {
      result = n;
      n = n->left();
    }
    else
    {
      CmiAssert(comp_result > 0);
      n = n->right();
    }
  }
  return result;
}

void isomempool::RBTree::replace_node(Node * oldn, Node * newn)
{
  Node * parent = oldn->parent();
  if (oldn->isRoot())
  {
    this->root = newn;
  }
  else
  {
    if (oldn == parent->left())
      parent->leftField = newn;
    else
      parent->rightField = newn;
  }
  if (newn != nullptr)
  {
    newn->parentField = oldn->parent();
  }
}

void isomempool::RBTree::rotate_left(Node * n)
{
  Node * r = n->right();
  replace_node(n, r);
  n->rightField = r->left();
  if (r->left() != nullptr)
  {
    r->left()->parentField = n;
  }
  r->leftField = n;
  n->parentField = r;
}
void isomempool::RBTree::rotate_right(Node * n)
{
  Node * L = n->left();
  replace_node(n, L);
  n->leftField = L->right();
  if (L->right() != nullptr)
  {
    L->right()->parentField = n;
  }
  L->rightField = n;
  n->parentField = L;
}

void isomempool::RBTree::insert_case1(Node * n)
{
  if (n->isRoot())
    n->setColor(BLACK);
  else
    insert_case2(n);
}
void isomempool::RBTree::insert_case2(Node * n)
{
  if (n->parent()->color() == BLACK)
    return; /* Tree is still valid */
  else
    insert_case3(n);
}
void isomempool::RBTree::insert_case3(Node * n)
{
  if (node_color(n->parent_sibling()) == RED)
  {
    n->parent()->setColor(BLACK);
    n->parent_sibling()->setColor(BLACK);
    n->grandparent()->setColor(RED);
    insert_case1(n->grandparent());
  }
  else
  {
    insert_case4(n);
  }
}
void isomempool::RBTree::insert_case4(Node * n)
{
  Node * parent = n->parent();
  if (n == parent->right() && parent == n->grandparent()->left())
  {
    rotate_left(parent);
    n = n->left();
  }
  else if (n == parent->left() && parent == n->grandparent()->right())
  {
    rotate_right(parent);
    n = n->right();
  }
  insert_case5(n);
}
void isomempool::RBTree::insert_case5(Node * n)
{
  Node * parent = n->parent();
  parent->setColor(BLACK);
  n->grandparent()->setColor(RED);
  if (n == parent->left() && parent == n->grandparent()->left())
  {
    rotate_right(n->grandparent());
  }
  else
  {
    CmiAssert(n == parent->right() && parent == n->grandparent()->right());
    rotate_left(n->grandparent());
  }
}

void isomempool::RBTree::insert(Node * inserted_node)
{
  if (this->root == nullptr)
  {
    this->root = inserted_node;
    inserted_node->parentField = nullptr;
    inserted_node->setColor(BLACK);
  }
  else
  {
    Node * n = this->root;
    while (1)
    {
      signed long long comp_result = inserted_node->key() - n->key();
      if (comp_result < 0)
      {
        if (n->left() == nullptr)
        {
          n->leftField = inserted_node;
          break;
        }
        else
        {
          n = n->left();
        }
      }
      else
      {
        if (n->right() == nullptr)
        {
          n->rightField = inserted_node;
          break;
        }
        else
        {
          n = n->right();
        }
      }
    }
    inserted_node->parentField = n;
    insert_case2(inserted_node);
  }

#ifdef VERIFY_RBTREE
  this->verify_properties();
#endif
}

void isomempool::RBTree::delete_case1(Node * n)
{
  if (n->isRoot())
    return;
  else
    delete_case2(n);
}
void isomempool::RBTree::delete_case2(Node * n)
{
  if (node_color(n->sibling()) == RED)
  {
    Node * parent = n->parent();
    parent->setColor(RED);
    n->sibling()->setColor(BLACK);
    if (n == parent->left())
      rotate_left(parent);
    else
      rotate_right(parent);
  }
  delete_case3(n);
}
void isomempool::RBTree::delete_case3(Node * n)
{
  Node * parent = n->parent();
  if (parent->color() == BLACK && node_color(n->sibling()) == BLACK &&
      node_color(n->sibling()->left()) == BLACK && node_color(n->sibling()->right()) == BLACK)
  {
    n->sibling()->setColor(RED);
    delete_case1(parent);
  }
  else
    delete_case4(n);
}
void isomempool::RBTree::delete_case4(Node * n)
{
  Node * parent = n->parent();
  if (parent->color() == RED && node_color(n->sibling()) == BLACK &&
      node_color(n->sibling()->left()) == BLACK && node_color(n->sibling()->right()) == BLACK)
  {
    n->sibling()->setColor(RED);
    parent->setColor(BLACK);
  }
  else
    delete_case5(n);
}
void isomempool::RBTree::delete_case5(Node * n)
{
  Node * parent = n->parent();
  if (n == parent->left() && node_color(n->sibling()) == BLACK &&
      node_color(n->sibling()->left()) == RED && node_color(n->sibling()->right()) == BLACK)
  {
    n->sibling()->setColor(RED);
    n->sibling()->left()->setColor(BLACK);
    rotate_right(n->sibling());
  }
  else if (n == parent->right() && node_color(n->sibling()) == BLACK &&
           node_color(n->sibling()->right()) == RED && node_color(n->sibling()->left()) == BLACK)
  {
    n->sibling()->setColor(RED);
    n->sibling()->right()->setColor(BLACK);
    rotate_left(n->sibling());
  }
  delete_case6(n);
}
void isomempool::RBTree::delete_case6(Node * n)
{
  Node * parent = n->parent();
  n->sibling()->setColor(parent->color());
  parent->setColor(BLACK);
  if (n == parent->left())
  {
    CmiAssert(node_color(n->sibling()->right()) == RED);
    n->sibling()->right()->setColor(BLACK);
    rotate_left(parent);
  }
  else
  {
    CmiAssert(node_color(n->sibling()->left()) == RED);
    n->sibling()->left()->setColor(BLACK);
    rotate_right(parent);
  }
}

void isomempool::RBTree::remove(Node * n)
{
  auto remove_internal = [this](Node * n)
  {
    CmiAssert(n->left() == nullptr || n->right() == nullptr);

    Node * child = n->right() == nullptr ? n->left() : n->right();
    if (node_color(n) == BLACK)
    {
      n->setColor(node_color(child));
      delete_case1(n);
    }
    replace_node(n, child);
    if (n->isRoot() && child != nullptr)  // root should be black
      child->setColor(BLACK);

#ifdef VERIFY_RBTREE
    this->verify_properties();
#endif
  };

  if (n->left() != nullptr && n->right() != nullptr)
  {
    /* Copy key/value from predecessor and then delete it instead */
    Node * pred = n->left()->maximum_node();
    remove_internal(pred);
    replace_node(n, pred);
    pred->setColor(n->color());
    pred->leftField = n->leftField;
    pred->rightField = n->rightField;
    if (pred->left())
      pred->left()->parentField = pred;
    if (pred->right())
      pred->right()->parentField = pred;
  }
  else
  {
    remove_internal(n);
  }
}

#ifdef VERIFY_RBTREE
void isomempool::RBTree::verify_property_1(Node * n)
{
  CmiAssert(node_color(n) == RED || node_color(n) == BLACK);
  if (n == nullptr) return;
  verify_property_1(n->left());
  verify_property_1(n->right());
}
void isomempool::RBTree::verify_property_2(Node * root)
{
  CmiAssert(node_color(root) == BLACK);
}
void isomempool::RBTree::verify_property_4(Node * n)
{
  if (node_color(n) == RED)
  {
    CmiAssert(node_color(n->left()) == BLACK);
    CmiAssert(node_color(n->right()) == BLACK);
    CmiAssert(n->parent()->color() == BLACK);
  }
  if (n == nullptr) return;
  verify_property_4(n->left());
  verify_property_4(n->right());
}
void isomempool::RBTree::verify_property_5_helper(Node * n, int black_count, int * path_black_count)
{
  if (node_color(n) == BLACK)
  {
    black_count++;
  }
  if (n == nullptr)
  {
    if (*path_black_count == -1)
    {
      *path_black_count = black_count;
    }
    else
    {
      CmiAssert(black_count == *path_black_count);
    }
    return;
  }
  verify_property_5_helper(n->left(), black_count, path_black_count);
  verify_property_5_helper(n->right(), black_count, path_black_count);
}
void isomempool::RBTree::verify_property_5(Node * root)
{
  int black_count_path = -1;
  verify_property_5_helper(root, 0, &black_count_path);
}

void isomempool::RBTree::verify_properties()
{
  verify_property_1(this->root);
  verify_property_2(this->root);
  /* Property 3 is implicit */
  verify_property_4(this->root);
  verify_property_5(this->root);
}
#endif
#endif

/************** External interface ***************/
#if USE_ISOMEMPOOL
static CmiIsomallocBlock *isomalloc_internal_alloc_block(size_t size, isomempool *pool)
{
  CmiIsomallocBlock *blk = (CmiIsomallocBlock*)pool->alloc(size);
  blk->start = blk;
  return blk;
}
static CmiIsomallocBlock *isomalloc_internal_alloc_block_aligned(size_t size, size_t align, size_t align_offset, isomempool *pool)
{
  CmiIsomallocBlock *blk = (CmiIsomallocBlock*)pool->alloc(size, align, align_offset);
  blk->start = blk;
  return blk;
}
#elif CMK_USE_MEMPOOL_ISOMALLOC
static CmiIsomallocBlock *isomalloc_internal_alloc_block(size_t size, mempool_type *pool)
{
  CmiIsomallocBlock *blk = (CmiIsomallocBlock*)mempool_malloc(pool, size, 1);
  blk->start = blk;
  return blk;
}
#else
static CmiIsomallocBlock *isomalloc_internal_alloc_block(size_t size)
{
  CmiInt8 s,n,i;
  CmiIsomallocBlock *blk;
  if (isomallocStart==NULL) return (CmiIsomallocBlock *)disabled_map(size);
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
  return blk;
}
#endif

#if USE_ISOMEMPOOL
void* CmiIsomallocFromIsomempool(size_t size, isomempool *pool)
{
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block(size + sizeof(CmiIsomallocBlock), pool);
  return block2pointer(blk);
}
#elif CMK_USE_MEMPOOL_ISOMALLOC
void* CmiIsomallocFromPool(size_t size, mempool_type *pool)
{
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block(size + sizeof(CmiIsomallocBlock), pool);
  return block2pointer(blk);
}
#else
void *CmiIsomallocPlain(int size)
{
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block(size + sizeof(CmiIsomallocBlock));
  blk->length = size;
  blk->align = 0;
  blk->alignoffset = 0;
  return block2pointer(blk);
}
#endif

#define MALLOC_ALIGNMENT           ALIGN_BYTES
#define MINSIZE                    (sizeof(CmiIsomallocBlock))

static size_t isomalloc_internal_validate_align(size_t align)
{
  if (align < MINSIZE) align = MINSIZE;
  /* make sure alignment is power of 2 */
  if ((align & (align - 1)) != 0) {
    size_t a = MALLOC_ALIGNMENT * 2;
    while ((unsigned long)a < (unsigned long)align) a <<= 1;
    return a;
  }
  return align;
}

static void *isomalloc_internal_perform_alignment(CmiIsomallocBlock *blk, size_t align, size_t alignoffset)
{
  void *ptr = block2pointer(blk);
  CmiIntPtr ptr2align = (CmiIntPtr)ptr + alignoffset;
  if (ptr2align % align != 0) { /* misaligned */
    CmiIsomallocBlock savedblk = *blk;
    ptr2align = (ptr2align + align - 1) & -((CmiInt8) align);
    ptr2align -= alignoffset;
    ptr = (void*)ptr2align;
    blk = pointer2block(ptr);      /* restore block */
    *blk = savedblk;
  }
  return ptr;
}

/* alignment occurs after blocklistsize bytes */
static void *isomalloc_internal_alloc_aligned(size_t useralign, size_t usersize, size_t blocklistsize, CmiIsomallocBlockList *list)
{
  size_t size = usersize + blocklistsize;
  size_t align = isomalloc_internal_validate_align(useralign);

#if USE_ISOMEMPOOL
  size_t align_offset = sizeof(CmiIsomallocBlock) + blocklistsize;
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block_aligned(size + sizeof(CmiIsomallocBlock), align, align_offset, list->pool);
  return block2pointer(blk);
#elif CMK_USE_MEMPOOL_ISOMALLOC
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block(size + sizeof(CmiIsomallocBlock) + align, list->pool);
  return isomalloc_internal_perform_alignment(blk, align, blocklistsize);
#else
  CmiIsomallocBlock *blk = isomalloc_internal_alloc_block(size + sizeof(CmiIsomallocBlock) + align);
  blk->length = size;
  blk->align = align;
  blk->alignoffset = blocklistsize;
  return isomalloc_internal_perform_alignment(blk, align, blocklistsize);
#endif
}

int CmiIsomallocEnabled(void)
{
  return (isomallocStart!=NULL);
}

#if !CMK_USE_MEMPOOL_ISOMALLOC && !USE_ISOMEMPOOL
void CmiIsomallocPup(pup_er p,void **blockPtrPtr)
{
  CmiIsomallocBlock *blk;
  CmiInt8 s,length,align,alignoffset;
  CmiInt8 n;
  if (isomallocStart==NULL) CmiAbort("isomalloc is disabled-- cannot use IsomallocPup");

  if (!pup_isUnpacking(p)) 
  { /*We have an existing block-- unpack start slot & length*/
    blk=pointer2block(*blockPtrPtr);
    s=blk->slot;
    length=blk->length;
    align=blk->align;
    alignoffset=blk->alignoffset;
  }

  pup_int8(p,&s);
  pup_int8(p,&length);
  pup_int8(p,&align);
  pup_int8(p,&alignoffset);
  n=length2slots(length + sizeof(CmiIsomallocBlock) + align);

  if (pup_isUnpacking(p)) 
  { /*Must allocate a new block in its old location*/
    if (pup_isUserlevel(p) || pup_isRestarting(p))
    {	/*Checkpoint: must grab old slots (even remote!)*/
      all_slotOP(&grabOP,s,n);
    }
    blk=map_slots(s,n);
    if (!blk) map_failed(s,n);
    blk->slot=s;

    blk->length = length;
    blk->align = align;
    blk->alignoffset = alignoffset;
    *blockPtrPtr = align > 0 ? isomalloc_internal_perform_alignment(blk, align, alignoffset) : block2pointer(blk);
  }

  /*Pup the allocated data*/
  pup_bytes(p,*blockPtrPtr,length);

  if (pup_isDeleting(p)) 
  { /*Unmap old slots, but do not mark as free*/
    unmap_slots(s,n);
    *blockPtrPtr=NULL; /*Zero out user's pointer*/
  }
}
#endif

#if USE_ISOMEMPOOL
static void CmiIsomallocFree(isomempool *pool, void *blockPtr)
#else
static void CmiIsomallocFree(void *blockPtr)
#endif
{
  if (isomallocStart==NULL) {
    disabled_unmap(pointer2block(blockPtr));
  }
  else if (blockPtr!=NULL)
  {
#if USE_ISOMEMPOOL
    pool->free(pointer2block(blockPtr)->start);
#elif CMK_USE_MEMPOOL_ISOMALLOC
    mempool_free_thread(pointer2block(blockPtr)->start);
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

int _sync_iso_warned = 0;

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
    /* Warn user if ASLR is enabled and '+isomalloc_sync' is missing */
    if (CmiMyPe() == 0 && read_randomflag() == 1 && _sync_iso == 0 && _sync_iso_warned == 0) {
      _sync_iso_warned = 1;
      CmiPrintf("Warning> Randomization of virtual memory (ASLR) is turned "
        "on in the kernel, thread migration may not work! Run 'echo 0 > "
        "/proc/sys/kernel/randomize_va_space' as root to disable it, "
        "or try running with '+isomalloc_sync'.\n");
    }

    init_ranges(argv);
  }
#endif
}

/***************** BlockList interface *********
  This was moved here from memory-isomalloc.C when it
  was realized that a list-of-isomalloc'd-blocks is useful for
  more than just isomalloc heaps.
  */

/*Convert a slot to a user address*/
static char *Slot_toUser(CmiIsomallocBlockList *s) {return (char *)(s+1);}
static CmiIsomallocBlockList *Slot_fmUser(void *s) {return ((CmiIsomallocBlockList *)s)-1;}

/*Build a new blockList.*/
CmiIsomallocBlockList *CmiIsomallocBlockListNew(void)
{
  CmiIsomallocBlockList *ret;

#if USE_ISOMEMPOOL
  isomempool *pool = new isomempool{};
  ret = (CmiIsomallocBlockList *)CmiIsomallocFromIsomempool(sizeof(*ret), pool);
  ret->pool = pool;
#elif CMK_USE_MEMPOOL_ISOMALLOC
  mempool_type *pool = mempool_init(2*(sizeof(CmiIsomallocBlock)+sizeof(mempool_header)) + sizeof(mempool_type),
                                    isomallocfn, isofreefn, 0);
  ret = (CmiIsomallocBlockList *)CmiIsomallocFromPool(sizeof(*ret), pool);
  ret->pool = pool;
#else
  ret = (CmiIsomallocBlockList *)CmiIsomallocPlain(sizeof(*ret));
#endif

  ret->next=ret; /*1-entry circular linked list*/
  ret->prev=ret;
  return ret;
}

/* BIGSIM_OOC DEBUGGING */
static void print_myslots(void);

#if USE_ISOMEMPOOL
void CmiIsomallocBlockListPup(pup_er cpup,CmiIsomallocBlockList **lp)
{
  PUP::er & p = *(PUP::er *)cpup;

  isomempool *pool;
  unsigned char dopup;
  if (!p.isUnpacking())
  {
    CmiAssert(*lp);
    pool = (*lp)->pool;
    dopup = pool != nullptr;
  }

  p | dopup;
  if (!dopup)
    return;

  if (p.isUnpacking())
    pool = new isomempool{};

  p | *pool;

  p((char *)lp, sizeof(void *));

  if (p.isUnpacking())
  {
    CmiIsomallocBlockList *head = *lp, *node = head;
    do
    {
      node->pool = pool;
      node = node->next;
    }
    while (node != head);
  }

  if (p.isDeleting())
  {
    delete pool;
    *lp = nullptr;
  }
}
#elif CMK_USE_MEMPOOL_ISOMALLOC
/*Pup all the blocks in this list.  This amounts to two circular
  list traversals.  Because everything's isomalloc'd, we don't even
  have to restore the pointers-- they'll be restored automatically!
  */
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **lp)
{
  mempool_type *mptr;
  block_header *current, *block_head;
  large_block_header* lcurr;
  slot_header *currSlot;
  void *newblock;
  CmiInt8 slot;
  size_t size;
  int flags[2];
  int i, j;
  int dopup = 1;
  int numBlocks = 0, numSlots = 0, flag = 1;

  if(!pup_isUnpacking(p)) {
    CmiAssert(*lp);
    mptr = (*lp)->pool;

    if(mptr == NULL) {
      dopup = 0;
    } else {
      dopup = 1;
    }
  }
  
  pup_int(p,&dopup);
  if(!dopup)  return;

  DEBUG_PRINT("[%d] My rank is %lld Pupping with isUnpack %d isDelete %d \n",
              CmiMyPe(),CthSelf(),pup_isUnpacking(p),pup_isDeleting(p));
  flags[0] = 0; flags[1] = 1;
  if(!pup_isUnpacking(p)) {
    current = MEMPOOL_GetBlockHead(mptr);
    while(current != NULL) {
      numBlocks++;
      current = MEMPOOL_GetBlockNext(current)?(block_header *)((char*)mptr+MEMPOOL_GetBlockNext(current)):NULL;
    }
    DEBUG_PRINT("Number of blocks packed %d\n",numBlocks);
    pup_int(p,&numBlocks);
    current = MEMPOOL_GetBlockHead(mptr);
    while(current != NULL) {
      pup_size_t(p,&(MEMPOOL_GetBlockSize(current)));
      pup_int8(p,(CmiInt8*)&(MEMPOOL_GetBlockMemHndl(current)));
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
        pup_size_t(p,&currSlot->size);
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
    //pup large blocks
    numBlocks = 0;
    lcurr = (mptr->large_blocks)?(large_block_header*)((char*)mptr + mptr->large_blocks):NULL;
    while(lcurr != NULL) {
      numBlocks++;
      lcurr = lcurr->block_next ? (large_block_header *)((char*)mptr + lcurr->block_next) : NULL;
    }
    pup_int(p,&numBlocks);
    lcurr = (mptr->large_blocks)?(large_block_header*)((char*)mptr + mptr->large_blocks):NULL;
    while(lcurr != NULL) {
      pup_size_t(p,&(MEMPOOL_GetBlockSize(lcurr)));
      pup_int8(p,(CmiInt8*)&(MEMPOOL_GetBlockMemHndl(lcurr)));
      pup_bytes(p,lcurr,MEMPOOL_GetBlockSize(lcurr));
      lcurr = lcurr->block_next ? (large_block_header *)((char*)mptr + lcurr->block_next) : NULL;
    }
  }

  if(pup_isUnpacking(p)) {
    //unpack regular blocks
    pup_int(p,&numBlocks);
    DEBUG_PRINT("Number of blocks to be unpacked %d\n",numBlocks);
    for(i = 0; i < numBlocks; i++) { 
      pup_size_t(p,&size);
      pup_int8(p,&slot);
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
        pup_size_t(p,&size);
        pup_int(p,&flags[1]);
        if(flags[1] == 0) {
          pup_bytes(p,newblock,sizeof(slot_header));
        } else {
          pup_bytes(p,newblock,size);
        }
        newblock = (char*)newblock + size;
      }
    }
    //unpack large blocks
    pup_int(p,&numBlocks);
    for(i = 0; i < numBlocks; i++) {
      pup_size_t(p,&size);
      pup_int8(p,&slot);
      newblock = map_slots(slot,size/slotsize);
      pup_bytes(p,newblock,size);
    }
#if (CMK_USE_MEMPOOL_ISOMALLOC && !USE_ISOMEMPOOL) || (CMK_SMP && CMK_CONVERSE_UGNI)
    mptr->mempoolLock = CmiCreateLock();
#endif  
  }
  pup_pointer(p,(void**)lp);
  if(pup_isDeleting(p)) {
    mempool_destroy(mptr);
    *lp=NULL;
  }
#if ISOMALLOC_DEBUG
    CmiPrintf("Isomalloc:pup done\n");
#endif
}
#else
void CmiIsomallocBlockListPup(pup_er p,CmiIsomallocBlockList **lp)
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
#if USE_ISOMEMPOOL
  isomempool *pool = cur->pool;
#elif CMK_USE_MEMPOOL_ISOMALLOC
  mempool_type *pool = cur->pool;
#endif
  do {
    CmiIsomallocBlockList *doomed=cur;
    cur=cur->next; /*Have to stash next before deleting cur*/
#if USE_ISOMEMPOOL
    CmiIsomallocFree(pool, doomed);
#else
    CmiIsomallocFree(doomed);
#endif
  } while (cur!=start);
#if USE_ISOMEMPOOL
# if ISOMEMPOOL_DEBUG
  pool->print_contents();
# endif
  delete pool;
#elif CMK_USE_MEMPOOL_ISOMALLOC
  mempool_destroy(pool);
#endif
}

/*Allocate a block into this blockList*/
void *CmiIsomallocBlockListMalloc(CmiIsomallocBlockList *l,size_t nBytes)
{
  CmiIsomallocBlockList *n; /*Newly created slot*/
#if USE_ISOMEMPOOL
  n = (CmiIsomallocBlockList *)CmiIsomallocFromIsomempool(sizeof(CmiIsomallocBlockList)+nBytes, l->pool);
  n->pool = l->pool;
#elif CMK_USE_MEMPOOL_ISOMALLOC
  n = (CmiIsomallocBlockList *)CmiIsomallocFromPool(sizeof(CmiIsomallocBlockList)+nBytes, l->pool);
  n->pool = l->pool;
#else
  n = (CmiIsomallocBlockList *)CmiIsomallocPlain(sizeof(CmiIsomallocBlockList)+nBytes);
#endif
  /*Link the new block into the circular blocklist*/
  n->prev=l;
  n->next=l->next;
  l->next->prev=n;
  l->next=n;
  return Slot_toUser(n);
}

/*Allocate a block into this blockList with alignment */
void *CmiIsomallocBlockListMallocAlign(CmiIsomallocBlockList *l,size_t align,size_t nBytes)
{
  CmiIsomallocBlockList *n; /*Newly created slot*/
  n=(CmiIsomallocBlockList *)isomalloc_internal_alloc_aligned(align, nBytes, sizeof(CmiIsomallocBlockList), l);
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
#if USE_ISOMEMPOOL
  CmiIsomallocFree(n->pool, n);
#else
  CmiIsomallocFree(n);
#endif
}

/* BIGSIM_OOC DEBUGGING */
static void print_myslots(void) {
  CmiPrintf("[%d] my slot set=%p\n", CmiMyPe(), CpvAccess(myss));
  print_slots(CpvAccess(myss));
}
