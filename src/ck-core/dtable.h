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
 * Revision 2.2  1995-07-25 00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.4  1995/05/09  20:55:53  knauff
 * Added SetId method, plus surrounding #ifndef #define wrappers
 * for multiple includes.
 *
 * Revision 1.2  1994/11/11  05:31:25  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:46  brunner
 * Initial revision
 *
 ***************************************************************************/
/**************************************************************************/
/*                                                                        */
/*      Authors: Wayne Fenton, Balkrishna Ramkumar, Vikram A. Saletore    */
/*                    Amitabh B. Sinha  and  Laxmikant V. Kale            */
/*              (C) Copyright 1990 The Board of Trustees of the           */
/*                          University of Illinois                        */
/*                           All Rights Reserved                          */
/*                                                                        */
/**************************************************************************/

#ifndef _dtable_h_
#define _dtable_h_

#define TBL_WAITFORDATA 1
#define TBL_NOWAITFORDATA 2

#define TBL_REPLY 1
#define TBL_NOREPLY 2

#define TBL_WAIT_AFTER_FIRST 1
#define TBL_NEVER_WAIT 2
#define TBL_ALWAYS_WAIT 3

message TBL_MSG {
	int key;
	char *data;
} ;

extern int _CK_NumTables ;
extern "C" void TblInsert(int, int, int, void *, int, int, ChareIDType *, int);
extern "C" void TblDelete(int, int, int, int, ChareIDType *, int) ;
extern "C" void TblFind(int, int, int, int, ChareIDType *, int) ;


class table { /* top level distributed table object */
        int _CK_MyId ;
public:
        table()
        {       _CK_MyId = _CK_NumTables++ ;
                /* Table ids are assigned at run time unlike in CHARM.
                   _CK_NumTables is a global, defined in cplus_node_main.c */
        }
        void Insert(int key, void *data, int size_data, int EPid, ChareIDType cid, int option)
        {
            if ( CK_PE_SPECIAL(GetID_onPE(cid)) )
                ::TblInsert(_CK_MyId, -1, key, data, size_data, EPid, NULL, option) ;
            else
                ::TblInsert(_CK_MyId, -1, key, data, size_data, EPid, &cid, option) ;
        }

        void Delete(int key, int EPid, ChareIDType cid, int option)
        {
            if ( CK_PE_SPECIAL(GetID_onPE(cid)) )
                ::TblDelete(_CK_MyId, -1, key, EPid, NULL, option) ;
            else
                ::TblDelete(_CK_MyId, -1, key, EPid, &cid, option) ;
        }

        void Find(int key, int EPid, ChareIDType cid, int option)
        {
            if ( CK_PE_SPECIAL(GetID_onPE(cid)) )
                ::TblFind(_CK_MyId, -1, key, EPid, NULL, option) ;
            else
                ::TblFind(_CK_MyId, -1, key, EPid, &cid, option) ;
        }

	int GetId()
	{	return _CK_MyId ; }

	void SetId (int x) {
	     _CK_MyId = x;
	}

} ;

#endif
