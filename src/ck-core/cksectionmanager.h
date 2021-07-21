#ifndef _CK_SECTION_MANAGER_
#define _CK_SECTION_MANAGER_
#include "cksec.decl.h"

#include "cksection.h"
#include <unordered_map>

// placeholder, but I think this will be some base Proxy class
using SectionEntry = void*;

class _SectionInfo
{
public:
  template<class T>
  using LocalMemberContainer = std::vector<T>;

private:
  CkSectionInfo info;
  LocalMemberContainer<SectionEntry> localElements;
  // children in the spanning tree, but I think we will (for now) defer to Ckmulticast for the spanning tree.
  std::vector<int> childPEs;
public:
  using size_type = LocalMemberContainer<SectionEntry>::size_type;
  _SectionInfo(size_type localElementSize)
    : localElements{localElementSize} {}
  _SectionInfo()
    : localElements{0} {}

  void addLocalMember(SectionEntry member)
  {
    localElements.push_back(member);
  }
  template<typename InputIt>
  void addLocalMembers(InputIt begin, InputIt end)
  {
    localElements.insert(localElements.end(), begin, end);
  }
  void addChildPE(int pe)
  {
    childPEs.push_back(pe);
  }

  const LocalMemberContainer<SectionEntry>&
  getLocalMembers() {return localElements;}

};

class CkSectionManager : public CBase_CkSectionManager {
public:
  using SectionMapType = std::unordered_map<int, int>;
private:
  SectionMapType sections;
  int lastCounter = 0;

  ck::SectionID createSectionID();

  template<class OutputIt>
  void createSectionIDs(OutputIt dest, int n)
  {
    for(int _ = 0; _ < n; ++_)
      {
        *dest = createSectionID();
        ++dest;
      }
  }

public:

  CkSectionManager();
  // Create a single section containing the chares in the range
  // [begin, end), return a handle to it that can be referenced
  // returns SectionID, will be used to create CProxy_SectionXX
  // CProxy_SectionXX will have the SectionID as a member
  template<class SectionFn, class InputIt>
  ck::SectionID createSection(SectionFn fn, InputIt begin, InputIt end)
  {
    ck::SectionID newSectionID = createSectionID();
    _SectionInfo newSectionInfo{};

    for(auto x = begin; x != end; x++)
      {
        if(fn(*x))
          {
            newSectionInfo.addLocalMember(*x);
          }
      }
    // // should move it, not copy
    sections[newSectionID] = newSectionInfo;
  }


  // create std::distance(outputBegin, outputEnd) sections from the chares in the range
  // [begin, end).
  // writes the section IDs of the created sections to the range [outputBegin, outputEnd)
  template<class SectionFn, class InputIt, class OutputIt>
  void createSection(SectionFn fn, InputIt begin, InputIt end, OutputIt outputBegin, OutputIt outputEnd)
  {
    auto nSections = std::distance(outputEnd, outputBegin);
    ck::SectionID ids[nSections];
    createSectionIDs(ids, nSections);
    int idIdx = 0;

    for(auto x = outputBegin; x != outputEnd; ++x)
      {
        auto id = ids[idIdx];
        *x = id;
        idIdx++;

        sections[id] = _SectionInfo();
      }

    for(const auto x = begin; x != end; ++x)
      {
        // assumes each chare is in exactly one section, but that may not be the case.
        // From here we could: specify that SectionFn can generate a sequence of section IDs
        // or it can return a vector of section ID
        auto sectionNum = fn(*x);
        auto sectionID = ids[sectionNum];
        sections[sectionID].addLocalMember(*x);
      }

  }

  // combine sections in the range [begin, end) where
  // each item in the range is a proxy to a section that has already been created
  template<typename InputIt>
  ck::SectionID combineSections(InputIt begin,  InputIt end)
  {
    ck::SectionID newSectionID = createSectionID();
    _SectionInfo newSectionInfo{};

    for(auto x = begin; x != end; ++x)
      {
        const auto& members = *x.getLocalMembers();
        newSectionInfo.addLocalMembers(members.begin(), members.end());
      }
    sections[newSectionID] = newSectionInfo;
    return newSectionID;
  }

};

#endif // _CK_SECTION_MANAGER_
