# -static to avoid missing dynamic libs
CMK_LD="$CMK_LD "
CMK_LDXX="$CMK_LDXX "
CMK_LIBS="$CMK_LIBS -lbproc"
CMK_NATIVE_LIBS="$CMK_NATIVE_LIBS -lbproc"
CMK_BPROC=true
