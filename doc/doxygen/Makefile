
doxygen: 
	doxygen charm.conf

install:  doxygen
	/bin/rm -r -f /www/doxygen/charm.old
	/bin/mv -f /www/doxygen/charm /www/doxygen/charm.old
	/bin/mv html /www/doxygen/charm
	chmod  775 /www/doxygen/charm
	/bin/rm -r -f latex
