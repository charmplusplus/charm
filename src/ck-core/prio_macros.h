/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.2  1994/11/11  05:25:06  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:21  brunner
 * Initial revision
 *
 ***************************************************************************/
/************************************************************************/
/*									*/
/*			Malloc Macro definitions			*/
/*									*/
/************************************************************************/

#define PGEN_COPY_VECTOR(vector1_ptr, vector2_ptr)\
{\
	PVECTOR		*p_ptr, *c_ptr;\
	int		vector1_size;\
\
	p_ptr = vector1_ptr;\
	c_ptr = vector2_ptr;\
	if ( (*p_ptr >> 24) > 24 )\
		vector1_size = ( ((*p_ptr >> 24) - 25) >> 5 )  + 2;\
	else\
		vector1_size = 1;\
\
        for (;p_ptr < vector1_ptr+vector1_size; p_ptr++,c_ptr++)\
		*c_ptr = *p_ptr;\
}

#define PGEN_VECTOR_SIZE(ptr,size)\
{\
	if ( (*ptr >> 24) > 24 )\
		size = ( ((*ptr >> 24) - 25) >> 5 )  + 2;\
	else\
		size = 1;\
}

#define PGEN_VECTOR_LENGTH(ptr,length)\
{\
	length = (*ptr >> 24) & 0xff;\
}


#define PGEN_GET_NEW_PVECTOR(ptr1,ptr2,branch_bits,ptr2_rel_prio)\
{\
	unsigned int	all_zeros=0x0, all_ones=0xffffffff;\
	PVECTOR	*p_ptr, *c_ptr;\
	int	ptr1_size, ptr2_size;\
	int	shift;\
	int	ptr1_length, ptr2_length;\
\
	   ptr1_length =  *ptr1 >> 24;\
	   ptr2_length =  ptr1_length + branch_bits;\
\
	   if ( ptr2_length > 248 )\
		ptr2_length = 248;\
\
	   if ( ptr1_length > 24)\
		ptr1_size = ( (ptr1_length - 25) >> 5 )  + 2;\
	   else\
		ptr1_size = 1;\
\
	   if ( ptr2_length > 24)\
		ptr2_size = ( (ptr2_length - 25) >> 5 )  + 2;\
	   else\
		ptr2_size = 1;\
\
	   branch_bits = ptr2_length - ptr1_length;\
	   p_ptr = ptr1;\
	   c_ptr = ptr2;\
	   for(;p_ptr < ptr1+ptr1_size; p_ptr++,c_ptr++)\
		*c_ptr = *p_ptr;\
	   c_ptr = ptr2;\
	   *ptr2 = (*ptr2 & 0x0ffffff) | ((ptr2_length & 0x0ff) << 24 );\
\
	if ( ptr2_rel_prio > 0 )\
	{\
	   if  ( ptr1_size == 1 )\
	   {\
	     if ( ptr2_size == ptr1_size )\
	     {\
		/* clear and modify the first word of the child bit vector */\
		*ptr2 = ( *ptr2 & (all_ones << (24-ptr1_length)) )\
			| (ptr2_rel_prio << (24-ptr2_length) );\
	   	ptr2 = c_ptr;\
	     }\
	     else\
	     {\
		/* if the parent bit vector is on a word boundry */\
		if ( ((ptr1_length + 8) % 32) == 0)\
		{\
		   /* set child ptr to the last word of the child vector */\
		   ptr2 = ptr2 + (ptr2_size - 1);\
		   /* append the child priority to that word */\
		   *ptr2 = (*ptr2 & all_zeros)\
			   | (ptr2_rel_prio << (32-branch_bits));\
	   	ptr2 = c_ptr;\
		}/* ptr1_length +8 % 32 == 0 */\
		else\
		{ /* if the parent bit vector is NOT on a word boundry */\
		      /*append the child priority to the child word */\
		   *ptr2 = (*ptr2 & (all_ones << (24-ptr1_length)))\
		    	  | (ptr2_rel_prio >> (branch_bits-(24-ptr1_length)) );\
		   /* go to the next word */\
		   ptr2++;\
		   /* append the remaining priority bits to the next\
			child bit vector*/\
		   *ptr2 = (*ptr2 & all_zeros)\
		      | ( ptr2_rel_prio << (32-(branch_bits-(24-ptr1_length))) );\
	   	ptr2 = c_ptr;\
		}/* ptr1_length +8 % 32 != 0 */\
	     }\
	   }\
	   else\
	   {\
	     if (ptr2_size == ptr1_size)\
	     {\
		/* set child ptr to the last word of the child vector */\
		ptr2 = ptr2 + ptr2_size - 1;\
		shift = 32 - ((ptr1_length - 24) % 32) - branch_bits;\
		/* modify the last word of the child bit vector */\
		*ptr2 = (*ptr2 & (all_ones << (shift+branch_bits)) )\
			| (ptr2_rel_prio << shift );\
	   	ptr2 = c_ptr;\
	     }\
	     else\
	     {\
		/* if the parent bit vector is on a word boundry */\
		if ( ((ptr1_length + 8) % 32) == 0)\
		{\
		   /* set child ptr to the last word of the child vector */\
		   ptr2 = ptr2 + ptr2_size - 1;\
		   /* append the child priority to that word */\
		   *ptr2 = (*ptr2 & all_zeros)\
			   | (ptr2_rel_prio << (32-branch_bits));\
	   	ptr2 = c_ptr;\
		}/* ptr1_length +8 % 32 == 0 */\
		else\
		{ /* if the parent bit vector is NOT on a word boundry */\
		 /*set child ptr to the last BUT ONE word of the childvector*/\
		   ptr2 = ptr2 + ptr2_size - 2;\
		   shift = branch_bits - ( ((ptr1_size<<5) - 8) - ptr1_length);\
		   /*append the child priority to the child word */\
		   *ptr2 =(*ptr2 & (all_ones<<((ptr1_size<<5)-8-ptr1_length)))\
			   | (ptr2_rel_prio >> shift );\
		   /* go to the next word */\
		   ptr2++;\
		   /* append the remaining priority bits to the next\
			child bit vector*/\
		   *ptr2 = (*ptr2 & all_zeros) | ptr2_rel_prio << (32 - shift);\
	   	   ptr2 = c_ptr;\
		}/* ptr1_length +8 % 32 != 0 */\
	     }\
	   }\
	}\
}

#define	PGEN_IDENTICAL_PVECTOR(ptrP, new_priority_ptr)\
{\
	   int	ptrP_size;\
\
	   /* get vector size */\
	   if ( (*ptrP >> 24) > 24 )\
		ptrP_size = ( ((*ptrP >> 24) - 25) >> 5 )  + 2;\
	   else\
		ptrP_size = 1;\
\
	   new_priority_ptr = (PVECTOR *) CmiAlloc(sizeof(PVECTOR)*ptrP_size);\
	   /* copy the parent bit vector to child bit vector */\
	   PGEN_COPY_VECTOR(ptrP, new_priority_ptr);\
}


#define	CEIL_LOG2(m, k)\
{\
	int	j;\
\
	j = 020000000000;\
\
	if ( k == 0 )\
		m = 32;\
	else if ( k==1 )\
		m = 31;\
	else if ( k==2 )\
		m = 30;\
	else if ( k==3 )\
		m = 30;\
	else if ( k==4)\
		m = 29;\
	else if ( k==5)\
		m = 29;\
	else if ( k==6 )\
		m = 29;\
	else if ( k==7 )\
		m = 29;\
	else if ( k > 7)\
	for ( m=0; m < 32; m++ )\
	{\
		if ( (j & k) == 0 )\
			j = j >> 1;\
		else\
			break;\
	}\
\
	m = 32 - m;\
}




