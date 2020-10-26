include ../common.mk

MODULE = future

CK_HEADER_PATH =$(CDIR)/include/ck
HEADER = $(CK_HEADER_PATH)/$(MODULE).h 

all: $(CK_HEADER_PATH) $(HEADER)

$(CK_HEADER_PATH):
	mkdir -p $(CK_HEADER_PATH)

$(HEADER): $(MODULE).h
	cp -f future.h $(HEADER)

clean:
	rm -f $(HEADER)
