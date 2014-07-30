/**
  Multithread-friendly hashtable.
  
  This is inspired by Azul's Dr. Cliff Click's work:
  http://www.stanford.edu/class/ee380/Abstracts/070221_LockFreeHash.pdf

Dr. Orion Sky Lawlor, lawlor@alaska.edu, 2011-05-24 (Public Domain)
*/
#ifndef __OSL_HASHTABLE_MT_H
#define __OSL_HASHTABLE_MT_H

#include "osl/porthread.h" /* for locks */
#include <vector> /* used internally */

#ifndef DEBUG_HASH_COLLISIONS
#  define DEBUG_HASH_COLLISIONS /* empty */
#endif


/**
  Multithread-friendly hashtable.
  
  Multiple reader threads work fine without any locking.
  Writers go through a write lock, mostly for hashtable resize.
  
  KEY: data type used as index into hashtable.  
  	Must have a working operator== that works with volatile values.
	Must have a special "missingk" value (passed to the constructor)
	not equal to any other key.
  
  VALUE: data type stored in the hashtable.
    Must have a working and *atomic* assignment operator.
	Without atomic assignment, we would have to add reader locks.
	Typically, for big VALUE types you should store a pointer in the hashtable.
  
  KEY_TO_INDEX: function (or object) taking a KEY, and 
  	returning a size_t.
*/
template <class KEY, class VALUE, class KEY_TO_INDEX>
class hashtable_mt {
public:
	KEY _missingk; /* invalid key */
	VALUE _missingv; /* invalid value */
	KEY_TO_INDEX _key_to_index; /* return size_t given KEY */

	/* This is the datatype stored in the hashtable slots. */
	struct KEY_VALUE {
		KEY k;
		VALUE v;
		KEY_VALUE(const KEY &k_,const VALUE &v_) :k(k_), v(v_) {}
		KEY_VALUE() {}
	};
private:
	/**
	  The hashtable's slots are stored in this separate immutable class.
	  The slots, once allocated, are written to but never moved;
	  they are only deallocated once all readers are finished.
	*/
	class table_storage {
		/** This is the the total number of slots (including blanks) in the table. */
		size_t _nslots;
		
		/** This is the number of used slots in the table. */
		size_t _nused;
		
		/** This is the data in the hashtable */
		volatile KEY_VALUE *_slots;
	public:
		table_storage(size_t nslots,const KEY_VALUE &init_value) {
			_nslots=nslots;
			_nused=0;
			_slots=new KEY_VALUE[nslots];
			for (size_t i=0;i<_nslots;i++) {
				_slots[i].k=init_value.k;
				_slots[i].v=init_value.v;
			}
		}
		~table_storage() {
			_nslots=0;
			delete[] _slots;
		}
		
		/** Return the number of slots in this table. */
		inline size_t nslots() const {return _nslots;}
		
		/** Return the number of slots used in this table. */
		inline size_t nused() const {return _nused;}
		
		/** Increment the number of slots used in this table (MUST HOLD WRITELOCK!) */
		inline void used_more() {_nused++;}
		inline void used_less() {_nused--;}
		inline void used_none() {_nused=0;}
		
		/** Wrap this index around to the size of the table. */
		inline size_t wraparound(size_t index) const { return index & (_nslots-1); }
		
		/** Return this slot (must be in range) */
		volatile KEY_VALUE &lookup(size_t index) { return _slots[index]; }
		const volatile KEY_VALUE &lookup(size_t index) const { return _slots[index]; }

                inline void reset(const KEY_VALUE &init_value){
                  for (size_t i=0;i<_nslots;i++) {
                    _slots[i].k=init_value.k;
                    _slots[i].v=init_value.v;
                  }
                }
	};
	inline table_storage *makestorage(size_t nslots) const {
		size_t next_power_two=16;
		while (next_power_two<nslots) next_power_two*=2;
		return new table_storage(next_power_two,KEY_VALUE(_missingk,_missingv));
	}

	/** The current storage can be *read* without acquiring a lock. */
	table_storage *_storage;
	
	/** All writes or accesses to the data below MUST hold this lock! */
	porlock _writelock;
	
		/** Write a copy of VALUE for this KEY, in this storage.
			MUST have the writelock held! */
		void write_to_storage(table_storage *s,const KEY &k,const VALUE &v);
		VALUE &single_write_to_storage(table_storage *s,const KEY &k, VALUE &v);
		
		/** Re-hash the current _storage to be bigger. 
			Can only be called with the writelock held. */
		void rehash(void) {
			table_storage *ns=makestorage(2*_storage->nslots());
			enumerator e=make_enumerator();
			const volatile KEY_VALUE *kv; while (NULL!=(kv=e.next())) {
				KEY_VALUE kv2; 
				kv2.k=kv->k;  
				kv2.v=kv->v; /* make copies, to get rid of volatile */
				write_to_storage(ns,kv2.k,kv2.v);
			}
			std::swap(ns,_storage); /* atomically swap in new storage */
			_storage_archive.push_back(ns); /* keep old storage around until readers done */
		}
	
		/** To avoid multithreaded race conditions, we can't immediately delete
		    storage during rehashing, so we archive it here. 
			This is only a constant factor of additional storage.
		*/
		std::vector<table_storage *> _storage_archive;
	
	void operator=(const hashtable_mt &) {} /* do not copy or swap us */
public:
	/** Read the VALUE for this KEY, or _missingv if the key is not present.
	    This is an amortized constant-time operation, and involves zero locks.
		Actual performance is proportional to the number of hashtable collisions.
	*/
	volatile const VALUE &get(const KEY &k) const;

	/** Return the number of valid values currently in this table.
		This is a truly constant time operation. */
	size_t size(void) {
		return _storage->nused();
	}
	
	/** This enumerator will return all the valid values in this table. 
	  The idiomatic usage is:
		enumerator e=make_enumerator();
		const volatile KEY_VALUE *kv; 
		while (NULL!=(kv=e.next())) { ... use kv here ... }
	*/
	class enumerator {
		const hashtable_mt<KEY,VALUE,KEY_TO_INDEX> *h;
		const table_storage *s;
		size_t index;
	public:
		enumerator(const hashtable_mt<KEY,VALUE,KEY_TO_INDEX> *h_,
			const table_storage *s_) :h(h_), s(s_), index(0) {}
		
		/** Return the next key-value pair, or return 0 if no more exist. */
		const volatile KEY_VALUE *next(void) {
			while (index<s->nslots()) {
				const volatile KEY_VALUE &kv=s->lookup(index++);
				if (!(kv.k==h->_missingk)) {
					return &kv;
				}
			}
			return 0;
		}
	};
	enumerator make_enumerator(void) const {return enumerator(this,_storage);}
	
	/** Write a copy of VALUE for this KEY.
	  This method must acquire the write lock to complete.
	  Performance is still amortized constant time, though rehashing
	  will be required occasionally.
	*/
	void put(const KEY &k,const VALUE &v);
	VALUE &single_put(const KEY &k, VALUE &v);
	
	/** FIXME: erase. Needs either to rehash or to add a "tombstone" value. */


	/** Make a hashtable with this initial size. */
	hashtable_mt(size_t maxsize,KEY_TO_INDEX key_to_index,
		KEY _missingk_=KEY(), VALUE _missingv_=VALUE()) 
		:_missingk(_missingk_), _missingv(_missingv_),
		_key_to_index(key_to_index),
		_storage(makestorage(maxsize))
	{ }
	
	/**
	 To shut down the table, make *sure* all readers and writers are done using it first!
	*/
	~hashtable_mt() { 
		flush();
		delete _storage; _storage=0;
	}
	
	/** Reinitialize the table to this length.  Gives up all existing keys and values. */
	void reinit(size_t newsize) { 
		flush();
		delete _storage; _storage=makestorage(newsize);
	}
	
	/** Reclaim any unused storage (including previous table sizes).
	   This is only valid of there are zero live readers. */
	void flush(void) {
		porlock_scoped autolock(&_writelock);
		for (unsigned int i=0;i<_storage_archive.size();i++)
			delete _storage_archive[i];
		_storage_archive.resize(0);
	}

        inline void reset(){
          flush();
          porlock_scoped autolock(&_writelock);
          _storage->reset(KEY_VALUE(_missingk,_missingv));
          _storage->used_none(); 
        }
	
};


/** Read the VALUE for this KEY.  
	This is an amortized constant-time operation, and involves zero locks.
	Actual performance is proportional to the number of hashtable collisions.
*/
template <class KEY, class VALUE, class KEY_TO_INDEX>
inline volatile const VALUE &hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::
get(const KEY &k) const 
{
	const table_storage *s=_storage; /*<- atomic grab of table */
	size_t index=_key_to_index(k);
	while (true) { /* while we haven't found that key yet */
		index=s->wraparound(index);
		const volatile KEY_VALUE &kv=s->lookup(index);
		if (kv.k==k) { /* found existing key */
			return kv.v;
		}
		if (kv.k==_missingk) { /* missing key */
			return _missingv;
		}
		index++; /* else move down: keep looking! */
		DEBUG_HASH_COLLISIONS
	}
}

/** Write a copy of VALUE for this KEY */
template <class KEY, class VALUE, class KEY_TO_INDEX>
void hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::
put(const KEY &k,const VALUE &v) 
{
	porlock_scoped autolock(&_writelock);
	
	/* If more than 1/3 of the slots are in use, enlarge the table first. 
		
	   If the table gets even close to full, collisions build up quickly, 
	   which is terrible for performance.
	   
	   The complexities of rehashing with ongoing writes is why we 
	   just lock everything in this function.
	*/
	if (_storage->nslots()<3*_storage->nused()) rehash();
	
	write_to_storage(_storage,k,v);
}

/** Write a copy of VALUE for this KEY */
template <class KEY, class VALUE, class KEY_TO_INDEX>
VALUE &hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::
single_put(const KEY &k, VALUE &v) 
{
	porlock_scoped autolock(&_writelock);
	
	/* If more than 1/3 of the slots are in use, enlarge the table first. 
		
	   If the table gets even close to full, collisions build up quickly, 
	   which is terrible for performance.
	   
	   The complexities of rehashing with ongoing writes is why we 
	   just lock everything in this function.
	*/
	if (_storage->nslots()<3*_storage->nused()) rehash();
	
	return single_write_to_storage(_storage,k,v);
}

/** Write a copy of VALUE for this KEY, in this storage.
   MUST have the writelock held! */
template <class KEY, class VALUE, class KEY_TO_INDEX>
void hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::
write_to_storage(typename hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::table_storage *s,
	const KEY &k,const VALUE &v) 
{
	int index=_key_to_index(k);
	while (true) { /* while we haven't found that key yet */
		index =s->wraparound(index); /* ===   index %= _size; */
		volatile KEY_VALUE &kv=s->lookup(index);
		if (kv.k==k) { /* fast path: reuse old key */
		write_and_return:
			kv.v=v;
			return;
		}
		if (kv.k==_missingk) { /* claim this unused key */
			kv.k=k;
			s->used_more();
			goto write_and_return;
		}
		index++; /* move down: keep looking! */
		DEBUG_HASH_COLLISIONS
	}
}

/** Write a copy of VALUE for this KEY, in this storage.
   MUST have the writelock held! */
template <class KEY, class VALUE, class KEY_TO_INDEX>
VALUE & hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::
single_write_to_storage(typename hashtable_mt<KEY,VALUE,KEY_TO_INDEX>::table_storage *s,
	const KEY &k, VALUE &v) 
{
	int index=_key_to_index(k);
	while (true) { /* while we haven't found that key yet */
		index =s->wraparound(index); /* ===   index %= _size; */
                // FIXME - in the original version, this is marked volatile,
                // why is this so? 
		// volatile KEY_VALUE &kv=s->lookup(index);
                // at this point, the current thread will have acquired the
                // writelock, so that no other thread can update the entry
                // therefore, the queried kv pair shouldn't be volatile
		KEY_VALUE &kv = (KEY_VALUE &) s->lookup(index);
		if (kv.k==k) { /* fast path: reuse old key */
			return kv.v;
		}
		if (kv.k==_missingk) { /* claim this unused key */
			kv.k=k;
			s->used_more();
                        kv.v = v;
                        return v;
		}
		index++; /* move down: keep looking! */
		DEBUG_HASH_COLLISIONS
	}
}

#endif
