/**
 * @file mpiCollisionClient.h
 * @brief mpi collision library's Chare group client, 
 * collisions are delivered to it and sets the return pointer
 * @author Ehsan Totoni
 * @version 
 * @date 2012-10-10
 */

/*readonly*/ extern CProxy_MainCollide mainProxy;


class MpiCollisionClient: public collideClient {

	CollisionList* returnColls;
	public:
		MpiCollisionClient() {
		}
		virtual ~MpiCollisionClient(){}
		/* --------------------------------------------------------------------------*/
		/**
		 * @brief charm library delivers potential collisions here to client, it sets the pointer
		 *
		 * @param src
		 * @param step
		 * @param colls list of potential collisions
		 */
		/* ----------------------------------------------------------------------------*/
		virtual void collisions(ArrayElement *src, int step, CollisionList &colls){
			int size = colls.length();
			for (int c=0;c<size;c++) {
				returnColls->add(colls[c].A,colls[c].B);
			}
			CkCallback cb(CkReductionTarget(MainCollide, done), mainProxy);
			src->contribute(sizeof(int), &size, CkReduction::sum_int, cb);
		}
		void setResultPointer(CollisionList *&colls) {
			returnColls = colls =new CollisionList();
		}

};
