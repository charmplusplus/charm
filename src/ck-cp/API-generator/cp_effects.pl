#!/usr/bin/perl

# Generate forward declarations for all types of control point effects.
# each one takes a name and an association (arrayProxy and/or entryID).
# use C++ to overload the function names.


open(OUT_H, ">cp_effects.h");
open(OUT_CPP, ">cp_effects.cpp");



open(FILE, "cp_effects.txt");
while($line = <FILE>){
  chomp $line;
  if(length($line) > 0){
    $cp = $line;
    $funcdecls .= "\tvoid $cp(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);\n";
    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\");\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\");\n";
    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\", NoControlPointAssociation);\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\", NoControlPointAssociation);\n";

    $funcdefs .= "\t  void ControlPoint::EffectDecrease::$cp(std::string s, const ControlPoint::ControlPointAssociation &a){ }\n";
    $funcdefs .= "\t  void ControlPoint::EffectIncrease::$cp(std::string s, const ControlPoint::ControlPointAssociation &a){ }\n";

  }
}





print OUT_H <<EOF;
#include <string>
#include <set>
#include <ckarray.h>

namespace ControlPoint {
  class ControlPointAssociation{
  public:
    std::set<int> EntryID;
    std::set<int> ArrayGroupIdx;
      ControlPointAssociation(){
	// nothing here yet
      }
  };
  
  class ControlPointAssociatedEntry : public ControlPointAssociation {
    public :
	ControlPointAssociatedEntry(int epid) : ControlPointAssociation() {
	  EntryID.insert(epid);
	}    
  };
  
  class ControlPointAssociatedArray : public ControlPointAssociation {
  public:
    ControlPointAssociatedArray(CProxy_ArrayBase &a) : ControlPointAssociation() {
      CkGroupID aid = arraybase.ckGetArrayID();
      int groupIdx = aid.idx;
      ArrayGroupIdx.insert(groupIdx);
    }    
  };
  
  ControlPointAssociation NoControlPointAssociation;

EOF

print OUT_H "namespace EffectIncrease {\n";
print OUT_H $funcdecls;
print OUT_H "}\n\n";

print OUT_H "namespace EffectDecrease {\n";
print OUT_H $funcdecls;
print OUT_H "}\n";

print OUT_H "} //namespace ControlPoint \n";


print OUT_CPP <<EOF;
#include "cp_effects.h"
using namespace ControlPoint;
int main(){

EOF

print OUT_CPP "$funccalls";

print OUT_CPP <<EOF;
  return 0;
}

EOF

print OUT_CPP "$funcdefs";
