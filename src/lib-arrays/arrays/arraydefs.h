#ifndef ARRAYDEFS_H
#define ARRAYDEFS_H

#include <charm++.h>

typedef enum {false, true} boolean;

#define MessageIndex(mt)	_CK_msg_##mt
#define ChareIndex(ct)		_CK_chare_##ct##::_CK_charenum_##ct
#define EntryIndex(ct,ep,mt)	_CK_chare_##ct##::##ep##_##mt
#define ConstructorIndex(ct,mt)	_CK_chare_##ct##::##ct##_##mt

#define ALIGN8(x)	(int)(8*(((x)+7)/8))

typedef GroupIdType GroupIDType;
typedef int MessageIndexType;
typedef int ChareIndexType;
typedef int EntryIndexType;

#endif // ARRAYDEFS_H















