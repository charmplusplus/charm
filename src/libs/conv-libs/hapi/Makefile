CHARM_DIR = ../../../..
CHARMC = $(CHARM_DIR)/bin/charmc $(OPTS)
LIB_DIR = $(CHARM_DIR)/lib
NAME = hapi

all: $(LIB_DIR)/lib$(NAME).a

$(LIB_DIR)/lib$(NAME).a: $(NAME).o
	$(CHARMC) -o $@ $<
	rm -f $(CHARM_DIR)/include/$(NAME).h
	cp $(NAME).h $(CHARM_DIR)/include

$(NAME).o: $(NAME).cpp $(NAME).h
	$(CHARMC) -c $<

clean:
	rm -f *.o $(LIB_DIR)/lib$(NAME).a
