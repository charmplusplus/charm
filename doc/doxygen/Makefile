
doxygen: 
	doxygen charm.conf

web:  doxygen
	-/bin/rm -r -f /www/doxygen/charm.old
	-/bin/rm -r -f /www/doxygen/charm.old.nobackup
	/bin/mv html /www/doxygen/charm.new
	/bin/mv -f /www/doxygen/charm /www/doxygen/charm.old.nobackup
	/bin/mv /www/doxygen/charm.new /www/doxygen/charm
	chown -R kale:kale /www/doxygen/charm
	-chown -R kale:kale /www/doxygen/charm.old.nobackup
	chmod -R 775 /www/doxygen/charm
	-chmod -R 775 /www/doxygen/charm.old.nobackup
