CHARM_DIR = ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
LIB_DIR = $(CHARM_DIR)/lib
CUDA_DIR = $(CUDA_HOME)

NAME = hapi

all: $(LIB_DIR)/lib$(NAME).a

$(LIB_DIR)/lib$(NAME).a: $(NAME).o
	$(CHARMC) -o $@ $<
	cp $(NAME).h $(CHARM_DIR)/include

$(NAME).o: $(NAME).cpp $(NAME).h
	$(CHARMC) -c $< -I $(CUDA_DIR)/include

clean:
	rm -f *.o $(LIB_DIR)/lib$(NAME).a $(CHARM_DIR)/include/$(NAME).h
