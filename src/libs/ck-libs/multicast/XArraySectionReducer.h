#include "charm++.h"

#ifndef X_ARRAY_SECTION_REDUCER_H
#define X_ARRAY_SECTION_REDUCER_H

namespace ck {
    namespace impl {

/** Helper class to complete the last step in a cross-array reduction
 *
 * Simply buffers a bunch of reduction messages as they arrive via
 * subsection reductions and performs a final reduction on these to generate
 * the final reduced message which is passed on to the client.
 *
 * @note: Temporary entity meant to exist only until delegated cross-array
 * reductions are implemented more optimally.
 */
class XArraySectionReducer
{
    public:
        ///
        XArraySectionReducer(int _numSubSections, CkCallback *_finalCB)
            : numSubSections(_numSubSections), finalCB(_finalCB), numReceived(0)
        {
            CkAssert(numSubSections > 0);
            msgList = new CkReductionMsg*[numSubSections];
            memset( msgList, 0, numSubSections*sizeof(CkReductionMsg*) );
        }

        ///
        ~XArraySectionReducer()
        {
            delete finalCB;
            delete [] msgList;
        }

        /// Each subsection reduction message needs to be passed in here
        void acceptSectionContribution(CkReductionMsg *msg)
        {
            msgList[numReceived++] = msg;
            if (numReceived >= numSubSections)
                finalReducer();
        }

    private:
        /// Triggered after all subsections have completed their reductions
        void finalReducer()
        {
            // Get a handle on the reduction function for this message
            CkReduction::reducerFn f = CkReduction::reducerTable[ msgList[0]->reducer ].fn;
            // Perform an extra reduction step on all the subsection reduction msgs
            CkReductionMsg *finalMsg = (*f)(numSubSections, msgList);
            // Send the final reduced msg to the client
            finalCB->send(finalMsg);
            // Delete the subsection redn msgs, accounting for any msg reuse
            for (int i=0; i < numSubSections; i++)
                if (msgList[i] != finalMsg) delete msgList[i];
            // Reset the msg list and counters in preparation for the next redn
            memset( msgList, 0, numSubSections*sizeof(CkReductionMsg*) );
            numReceived = 0;
        }

        // The number of subsection redn msgs to expect
        const int numSubSections;
        // The final client callback after all redn are done
        const CkCallback *finalCB;
        // Counter to track when all subsection redns are complete
        int numReceived;
        // List of subsection redn msgs
        CkReductionMsg **msgList;
};


/// The reduction client that has to be registered for each subsection
void processSectionContribution (void *that, void *msg)
{
    CkAssert(that);
    reinterpret_cast<XArraySectionReducer*>(that)->acceptSectionContribution(reinterpret_cast<CkReductionMsg*>(msg));
}

    } // end namespace impl
} // end namespace ck

#endif // X_ARRAY_SECTION_REDUCER_H

