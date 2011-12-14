
#include "converse.h"
#include "mem-arena.h"

#if USE_BTREE

/* ======================================================================
 * New (b-tree-based implementation)
 * ====================================================================== */


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
  dllnode *new_dlln = (dllnode *)(malloc(sizeof(dllnode)));

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
  free(sb->listblock);

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
  btreenode *btn = (btreenode *)malloc(sizeof(btreenode));
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

  int index, inc;
  int i;

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
	  int i;
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

  {   /* BLOCK
     At this point, the desired slotblock has been removed, and we're
     going back up the tree.  We must check for deficient nodes that
     require the rotating or combining of elements to maintain a
     balanced b-tree. */
  int i;
  int def_child = -1;

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
    }    /* BLOCK */
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

      {   /* BLOCK */
      /* move the elements and children of the right child node to the */
      /* left child node */
      int num_left  = node->child[index]->num_blocks;
      int num_right = node->child[index+1]->num_blocks;
      int left_pos;
      int right_pos = 0;
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
      free(node->child[index+1]);
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
      }  /* BLOCK */
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
      free(node);
      node = new_root;
    }
  }

  return node;

}

/*****************************************************************
 * Creates a new slotset with nslots entries, starting with all empty
 * slots.  The slot numbers are [startslot, startslot + nslots - 1].
 *****************************************************************/

slotset *new_slotset(CmiInt8 startslot, CmiInt8 nslots) {
  int i;
  int list_bin;

  /* CmiPrintf("*** New Isomalloc ***\n"); */

  /* allocate memory for the slotset */
  slotset *ss = (slotset *)(malloc(sizeof(slotset)));

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
  ss->list_array[list_bin] = (dllnode *)(malloc(sizeof(dllnode)));
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

CmiInt8 get_slots(slotset *ss, CmiInt8 nslots) {

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

void grab_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots) {

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

void free_slots(slotset *ss, CmiInt8 sslot, CmiInt8 nslots) {

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
      free(node->child[i]);
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
	free(dlln->next);
      }
      free(dlln);
    }
  }
}

/*****************************************************************
 * Free the allocated memory of the slotset
 *****************************************************************/

static void delete_slotset(slotset *ss) {
  delete_btree(ss->btree_root);
  delete_list_array(ss);
  free(ss->btree_root);
  free(ss);
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


#else

/************************************************************************
     array-based memory allocator
************************************************************************/

/*
 * creates a new slotset of nslots entries, starting with all
 * empty slots. The slot numbers are [startslot,startslot+nslot-1]
 */

slotset *
new_slotset(CmiInt8 startslot, CmiInt8 nslots)
{
  /* CmiPrintf("*** Old isomalloc ***\n"); */
  int i;
  slotset *ss = (slotset*) malloc(sizeof(slotset));
  _MEMCHECK(ss);
  ss->maxbuf = 16;
  ss->buf = (slotblock *) malloc(sizeof(slotblock)*ss->maxbuf);
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

CmiInt8
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

void
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
    slotblock *newbuf = (slotblock *) malloc(sizeof(slotblock)*newsize);
    _MEMCHECK(newbuf);
    for (i=0; i<(ss->maxbuf); i++)
      newbuf[i] = ss->buf[i];
    for (i=ss->maxbuf; i<newsize; i++)
      newbuf[i].nslots  = 0;
    free(ss->buf);
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
void
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
void
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


#endif
