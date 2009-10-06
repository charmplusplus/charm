
doxygen: 
	doxygen charm.conf

web:  doxygen
	/bin/rm -r -f /www/doxygen/charm.old 
	/bin/mv -f /www/doxygen/charm /www/doxygen/charm.old
	/bin/mv html /www/doxygen/charm.new
	/bin/mv /www/doxygen/charm.new /www/doxygen/charm
	chown -R kale:kale /www/doxygen/charm  /www/doxygen/charm.old
	chmod -R 775 /www/doxygen/charm  /www/doxygen/charm.old
