CMK_CC_FLAGS="$CMK_CC_FLAGS -fsanitize=thread -fPIC"
CMK_CXX_FLAGS="$CMK_CXX_FLAGS -fsanitize=thread -fPIC"
CMK_LD_FLAGS="$CMK_LD_FLAGS -fsanitize=thread -pie"
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -fsanitize=thread -pie"
