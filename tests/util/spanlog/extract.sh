#!/bin/bash
# Prints the entry-span log's structs and functions from hapi_impl.cpp (argument 1), each from its first line to the
# first line that is a lone closing brace, for the host test to include.
F=$1
ext() { local N=$(grep -n "$1" $F | head -1 | cut -d: -f1); [ -n "$N" ] || { echo "#error missing: $1"; return; }
        awk -v s=$N 'NR>=s { print; if (/^\};?$/ || (NR==s && /\};$/)) exit }' $F; }
echo "// extracted from $F by extract.sh -- do not edit"
ext '^struct CuptiEntrySpan'
ext '^struct CuptiEntryLane'
ext '^static inline void hapiCuptiLaneOpen'
ext '^static inline void hapiCuptiLaneClose'
ext '^static void hapiCuptiLanePrune'
ext '^struct CuptiLaneCopy'
echo 'static std::atomic<uint64_t> g_cupti_snap_bytes{0}, g_cupti_snap_ns{0};'
ext '^static void hapiCuptiLaneRefresh'
ext '^static const CuptiEntrySpan\* hapiCuptiSpanAt'
