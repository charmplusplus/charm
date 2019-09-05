# bigsim:  bigsim-target

# bigemulator: bigsim

# bigsim-target: charm++ $(L)/libconv-bigsim.a $(L)/libconv-bigsim-logs.a $(L)/libconv-bluegene-stub.a $(L)/libblue-standalone.a

# $(L)/libbluegene.a: BlueGene.o
#     $(CHARMC) -o $@ $^

# # (Converse) Blue Gene emulator
# BLUE_OBJS = blue.o bigsim_node.o bigsim_proc.o bigsim_init.o bigsim_timing.o bigsim_record.o bigsim_ooc.o shared_alloc.o
# BIGSIM_IO_OBJS = bigsim_read.o bigsim_logs.o bigsim_api.o

# LIBBLUE_OBJS = $(BLUE_OBJS) $(BIGSIM_IO_OBJS) blue_stub.o blue_standalone.o

# $(L)/libconv-bigsim.a: $(BLUE_OBJS)
#     $(CHARMC) -o $@ $^

# $(L)/libconv-bigsim-logs.a: $(BIGSIM_IO_OBJS)
#     $(CHARMC) -o $@ $^

# $(L)/libconv-bluegene-stub.a:  blue_stub.o
#     $(CHARMC) -o $@ $^

# $(L)/libblue-standalone.a: blue_standalone.o
#     $(CHARMC) -o $@ $^


add_library(conv-bigsim blue.C bigsim_node.C bigsim_proc.C bigsim_init.C bigsim_timing.C bigsim_record.C bigsim_ooc.C shared_alloc.C)
add_library(conv-bigsim-logs bigsim_read.C bigsim_logs.C bigsim_api.C)
add_library(conv-bluegene-stub blue_stub.C)
add_library(blue-standalone blue_standalone.C)

add_dependencies(conv-bigsim ck)
add_dependencies(conv-bigsim-logs ck)
add_dependencies(conv-bluegene-stub ck)
add_dependencies(blue-standalone ck)
