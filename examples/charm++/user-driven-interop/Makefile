include Makefile.in

all: hello_user

hello_user: hello_user.cpp
	$(CHARMC) hello_user.ci
	$(CHARMC) -c hello_user.cpp -o hello_user

clean:
	rm -f hello_user *.o charmrun *.decl.h *.def.h

test: all

