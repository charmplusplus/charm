#  CRAY PERFTOOLS SUPPORT
#  =================
#  Perftools/6.4.3 or later is required for Charm++ support.
#  Do not load perftools or perftools-lite when building Charm++.
#    Instead, ensure frame pointers are not omitted in your conv-mach.sh
#       file.  (see below)
#  After Charm++ is built, load perftools-base and perftools before building
#    your application (such as NAMD). Then pat_build as desired.
#  Note there are perftools trace groups designed to help you study how Charm++
#    or Converse is used by your application.  Use pat_build -gcharm++ or
#    pat_build -gconverse.
#    See man pat_build.

# If building for Cray perftools, use the following.
#   Do not have perftools or perftools-lite loaded during
#     the build of charm++ (perftools-base is ok)
CRAYPAT_FLAGS=" "
if test -n "$PGCC"
then
    CRAYPAT_FLAGS=" "
elif test -n "$CCE"
then
    CRAYPAT_FLAGS=" -hkeep_frame_pointer"
elif test -n "$ICPC"
then
    CRAYPAT_FLAGS=" -fno-omit-frame-pointer"
else   # gcc
    CRAYPAT_FLAGS=" -fno-omit-frame-pointer"
fi

CMK_SEQ_CC_FLAGS="$CMK_SEQ_CC_FLAGS -fPIC"
CMK_SEQ_CXX_FLAGS="$CMK_SEQ_CXX_FLAGS -fPIC"

CMK_CC_FLAGS="$CMK_CC_FLAGS $CRAYPAT_FLAGS"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS $CRAYPAT_FLAGS"
