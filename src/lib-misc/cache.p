#include "cache.int"

module Cache {

#include "charm-inc.h"

#define INSERT 100
#define FIND 101

#define EMPTY 0
#define REQUESTED  1
#define HAVE  2

#define MAX 1009

#define CacheHash(key) ((1019*key) % MAX)

message {
	int x;
} InitMsg;

typedef struct _tbl_id_entry {
	int tbl, boc;
	struct _tbl_id_entry *next;
} TBL_ID_ENTRY;

typedef struct _address_entry {
	int ep;
	ChareIDType chareid;
	struct _address_entry *next;
} ADDRESS_ENTRY;

typedef struct _hash_entry {
	void *data;
	char status;
	int key;
	struct _address_entry *address;
	struct _hash_entry *next;
} HASH_ENTRY;


BranchOffice CacheTable {

	int tbl;
	ChareIDType bocid;
	HASH_ENTRY *hash_table[MAX];

	entry BranchInit: (message InitMsg *m) {
		int i;
		tbl = m->x;
		for (i=0; i<MAX; i++) hash_table[i] = NULL;
		MyBranchID(&bocid);
		CkFreeMsg(m);
	}



	private HASH_ENTRY * InsertHash(key)
	int key;
	{
		int index = CacheHash(key);
		HASH_ENTRY *temp = (HASH_ENTRY *) CkAlloc(sizeof(HASH_ENTRY));
		temp->key = key;
		temp->address = NULL;
		temp->next = hash_table[index];
		temp->data = NULL;
		temp->status = EMPTY;
		hash_table[index] = temp;
		return temp;
	}

	private HASH_ENTRY * FindHash(key)
	int key;
	{
		int index = CacheHash(key);
		HASH_ENTRY *temp = hash_table[index];
		while (temp) {
			if (temp->key==key) return temp;
			temp=temp->next;
		}
		return NULL;
	}

	private AddAddress(t, ep, id)
	HASH_ENTRY *t;
	int ep;
	ChareIDType *id;
	{
		ADDRESS_ENTRY * a = (ADDRESS_ENTRY *) CkAlloc(sizeof(ADDRESS_ENTRY));
		a->ep = ep;
		a->chareid = *id;
		a->next =  t->address;
		t->address = a;
	}

	public Find(key, ep, id, option)
	int key, ep; 
	ChareIDType *id;
	int option;
	{
		TBL_MSG *msg;
		HASH_ENTRY *t = PrivateCall(FindHash(key));

		if (!t) t = PrivateCall(InsertHash(key));
        switch (t->status) {

        case EMPTY:
			t->status = REQUESTED;
            PrivateCall(AddAddress(t, ep, id));
            TblFind(tbl, key, recv, &bocid, option);
            break;

        case REQUESTED:
			if (option == TBL_NEVER_WAIT) {
				msg = (TBL_MSG *) CkAllocMsg(TBL_MSG);
				msg->key = key;
				msg->data = NULL;
				SendMsg(ep, msg, id);
			}
			else
            	PrivateCall(AddAddress(t, ep, id));
            break;

        case HAVE:
			msg = (TBL_MSG *) CkAllocMsg(TBL_MSG);
			msg->key = key;
			msg->data = t->data;
			SendMsg(ep, msg, id);
            break;
        }

	}

	entry recv: (message TBL_MSG *msg) {

		ADDRESS_ENTRY *temp;
		HASH_ENTRY *t = PrivateCall(FindHash(msg->key));

		t->status = HAVE;
		t->data = msg->data;
		while (t->address) {
			TBL_MSG * m;

			temp = t->address;
			m = (TBL_MSG *) CkAllocMsg(InitMsg);
			m->key = msg->key;
			m->data = msg->data;
			SendMsg(t->address->ep, m, t->address->chareid);
			t->address = t->address->next;
			CkFree(temp);
		}
	}

}

OperateEntry(option, tbl, boc)
int option, tbl, *boc;
{
	TBL_ID_ENTRY *t;
	static TBL_ID_ENTRY  *id_table = NULL;

	switch (option) {
		case INSERT:
			t = (TBL_ID_ENTRY *) CkAlloc(sizeof(TBL_ID_ENTRY));
			t->tbl = tbl;
			t->boc = *boc;
			t->next = id_table;
			id_table = t;
			break;
			
		case FIND:
			t=id_table;
			while (t) { 
				if (t->tbl == tbl) {
					*boc = t->boc;
					break;
				}
				else t = t->next;
			}
			break;

		default:
			CkPrintf("*** ERROR *** Unknown option for OperateEntry.\n");
	}
}

Create(tbl)
int tbl;
{
	InitMsg *msg;
	int boc;

	msg = (InitMsg *) CkAllocMsg(InitMsg);
	msg->x = tbl;
	boc =  CreateBoc(CacheTable, CacheTable@BranchInit, msg);
	OperateEntry(INSERT, tbl, &boc);
}


Find(tbl, key, ep, id, option)
int tbl, key, ep;
ChareIDType *id;
int option;
{
	int boc;
	OperateEntry(FIND, tbl, &boc),
	BranchCall(boc, CacheTable@Find(key, ep, id, option));
}
}
