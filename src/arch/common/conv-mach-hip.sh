BUILD_HIP=1
if [ -n "$ROCM_PATH" ] && [ -d "$ROCM_PATH/include" ]; then
	CMK_ROCM_PATH="$ROCM_PATH"
elif [ -d "/opt/rocm/include" ]; then
	CMK_ROCM_PATH="/opt/rocm"
elif [ -d "/opt/rocm-default/include" ]; then
	CMK_ROCM_PATH="/opt/rocm-default"
elif [ -d "/opt/rocm-6.2.4/include" ]; then
	CMK_ROCM_PATH="/opt/rocm-6.2.4"
else
	CMK_ROCM_PATH="/opt/rocm"
fi

CMK_ROCM_LIBDIR="$CMK_ROCM_PATH/lib"
if [ ! -d "$CMK_ROCM_LIBDIR" ] && [ -d "$CMK_ROCM_PATH/lib64" ]; then
	CMK_ROCM_LIBDIR="$CMK_ROCM_PATH/lib64"
fi

CMK_INCDIR="-I$CMK_ROCM_PATH/include $CMK_INCDIR "
CMK_LIBDIR="-L$CMK_ROCM_LIBDIR $CMK_LIBDIR "
CMK_LIBS="-lhybridapi -lamdhip64 $CMK_LIBS "
