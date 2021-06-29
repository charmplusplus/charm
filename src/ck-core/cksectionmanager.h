#ifndef _CK_SECTION_MANAGER_
#define _CK_SECTION_MANAGER_
#include "cksectionmanager.def.h"

#include "cksection.h"
#include <unordered_map

class CkSectionManager : public CBase_CkSectionManager {
private:
  SectionMapType sections;

public:
  // temporary, don't know what this will actually be
  using SectionMapType = std::unordered_map<ck::SectionID, std::vector<int>>;

  // Create a single section containing the chares in the range
  // [begin, end), return a handle to it that can be referenced
  // returns SectionID, will be used to create CProxy_SectionXX
  // CProxy_SectionXX will have the SectionID as a member
  template<class SectionFn, class InputIt>
  ck::SectionID createSection(SectionFn, InputIt begin, InputIt end);


  // create std::distance(outputBegin, outputEnd) sections from the chares in the range
  // [begin, end). (Enforce that OutputIt is random access iterator
  template<class SectionFn, class InputIt, class OutputIt>
  ck::SectionID createSection(SectionFn, InputIt begin, InputIt end, OutputIt outputBegin, OutputIt outputEnd);

  // combine sections in the range [begin, end) where
  // each item in the range is a proxy to a section that has already been created
  template<typename InputIt>
  ck::SectionID combineSections(InputIt begin,  InputIt end);
};

#include "cksectionmanager.decl.h"

#endif // _CK_SECTION_MANAGER_
