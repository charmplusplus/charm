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
 * Revision 1.1  1995-06-13 11:32:16  jyelon
 * Initial revision
 *
 * Revision 1.1  1995/06/13  10:06:34  jyelon
 * Initial revision
 *
 * Revision 1.2  1994/11/11  05:20:00  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:37:44  brunner
 * Initial revision
 *
 ***************************************************************************/
#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IMaxRedn
#define GENERIC_REDOP_NAME Rmax 
#define GENERIC_OPERATOR  if (x[i] > *y) *y = x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 if (x[i] > y[i]) y[i] = x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IMinRedn
#define GENERIC_REDOP_NAME Rmin
#define GENERIC_OPERATOR  if (x[i] < *y) *y = x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 if (x[i] < y[i]) y[i] = x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME ISumRedn
#define GENERIC_REDOP_NAME Rsum
#define GENERIC_OPERATOR  *y += x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 y[i] += x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME IProdRedn
#define GENERIC_REDOP_NAME Rprod
#define GENERIC_OPERATOR  *y *= x[i];
#define GENERIC_OPERATOR1 y[i] = x[i];
#define GENERIC_OPERATOR2 y[i] *= x[i];

#define GENERIC_DATATYPE int
#define GENERIC_MODULE_NAME ICountRedn
#define GENERIC_REDOP_NAME Rcount
#define GENERIC_OPERATOR  (*y) += (x[i]==0) ? 0 : 1;
#define GENERIC_OPERATOR1 y[i]  = (x[i]==0) ? 0 : 1;
#define GENERIC_OPERATOR2 y[i] += (x[i]==0) ? 0 : 1;

