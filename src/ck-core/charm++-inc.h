/***** quiescence.h : included in user CHARM++ files. *****/

#ifndef _quiescence_h_
#define _quiescence_h_

message class QUIESCENCE_MSG {
public:	int msgs_processed;
} ;


#define TBL_WAITFORDATA 1
#define TBL_NOWAITFORDATA 2

#define TBL_REPLY 1
#define TBL_NOREPLY 2

#define TBL_WAIT_AFTER_FIRST 1
#define TBL_NEVER_WAIT 2
#define TBL_ALWAYS_WAIT 3


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
