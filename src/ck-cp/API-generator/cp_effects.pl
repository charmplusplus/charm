#!/usr/bin/perl

use strict;
use warnings;

# Generate forward declarations for all types of control point effects.
# each one takes a name and an association (arrayProxy and/or entryID).
# use C++ to overload the function names.

open(OUT_H, ">cp_effects.h");
open(OUT_CPP, ">cp_effects.C");
open(FILE, "cp_effects.txt");

my $funcdecls;
my $funccalls;
my $funcdefs;

while(my $line = <FILE>) {
  chomp $line;

  if(length($line) > 0) {
    my $cp = $line;

    $funcdecls .= "\tvoid $cp(std::string name, const ControlPoint::ControlPointAssociation &a = NoControlPointAssociation);\n";

    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\");\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\");\n";
    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\", NoControlPointAssociation);\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\", NoControlPointAssociation);\n";
    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\", EntryAssociation);\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\", EntryAssociation);\n";
    $funccalls .= "\tControlPoint::EffectIncrease::$cp(\"name\", ArrayAssociation);\n";
    $funccalls .= "\tControlPoint::EffectDecrease::$cp(\"name\", ArrayAssociation);\n";

    $funcdefs .= "void ControlPoint::EffectDecrease::$cp(std::string s, const ControlPoint::ControlPointAssociation &a) {\n" .
	"\tinsert(\"$cp\", s, a, EFF_DEC);\n" .
	"}\n";
    $funcdefs .= "void ControlPoint::EffectIncrease::$cp(std::string s, const ControlPoint::ControlPointAssociation &a) {\n" .
	"\tinsert(\"$cp\", s, a, EFF_INC);\n" .
	"}\n";
  }
}

print OUT_H <<EOF;
#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include "charm++.h"
#include "ck.h"
#include "ckarray.h"

namespace ControlPoint {
  class ControlPointAssociation {
  public:
    std::set<int> EntryID;
    std::set<int> ArrayGroupIdx;
      ControlPointAssociation() {
	// nothing here yet
      }
  };
  
  class ControlPointAssociatedEntry : public ControlPointAssociation {
    public :
	ControlPointAssociatedEntry() : ControlPointAssociation() {}

	ControlPointAssociatedEntry(int epid) : ControlPointAssociation() {
	  EntryID.insert(epid);
	}    
  };
  
  class ControlPointAssociatedArray : public ControlPointAssociation {
  public:
    ControlPointAssociatedArray() : ControlPointAssociation() {}

    ControlPointAssociatedArray(CProxy_ArrayBase &a) : ControlPointAssociation() {
      CkGroupID aid = a.ckGetArrayID();
      int groupIdx = aid.idx;
      ArrayGroupIdx.insert(groupIdx);
    }    
  };
  
  ControlPointAssociation NoControlPointAssociation;
  int epid = 2;
  ControlPointAssociatedEntry EntryAssociation(epid);
  ControlPointAssociatedArray ArrayAssociation;

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
using namespace std;

enum EFFECT {EFF_DEC, EFF_INC};

EOF

print OUT_CPP <<EOF;
typedef map<std::string, map<std::string, vector<pair<int, ControlPoint::ControlPointAssociation> > > > cp_effect_map;
typedef map<std::string, vector<pair<int, ControlPoint::ControlPointAssociation> > > cp_name_map;

CkpvDeclare(cp_effect_map, cp_effects);
CkpvDeclare(cp_name_map, cp_names);

void initControlPointEffects() {
\tCkpvInitialize(cp_effect_map, cp_effects);
\tCkpvInitialize(cp_name_map, cp_names);
}

void testControlPointEffects() {

EOF

print OUT_CPP "$funccalls";

print OUT_CPP <<EOF;
}

void insert(const std::string control_type, const std::string name, const ControlPoint::ControlPointAssociation &a, const int effect) {
\tCkpvAccess(cp_effects)[control_type][name].push_back(std::make_pair(effect, a));
\tCkpvAccess(cp_names)[name].push_back(std::make_pair(effect, a));
}
EOF

print OUT_CPP "$funcdefs";
