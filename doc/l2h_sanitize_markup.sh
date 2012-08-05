#!/bin/sh
#
# Shell script to fix Latex2HTML's broken absolute paths
# in image files, by replacing them with relative paths.
#
#  Orion Sky Lawlor, olawlor@acm.org, 2003/12/10

die() {
	echo $@
	exit 0
}

[ -x manual ] || die "this script requires a manual/ directory"

cp -r fig figs manual/

for f in `echo manual/*.html`
do
	echo "Converting $f"
	cwd=`pwd`
	cwd=`echo $cwd | sed -e 's@/home/net@/expand/home@'`
	sed -e 's!'`pwd`'/!!g' $f > tmp || die "error running sed on $f"
    # Uncomment to produce backup files for identifying the results of regex
	#mv $f $f.bak || die "error backing up $f"

    # Munge through the markup and... 
	# Relativize all paths
	# Replace placeholder with script tag
	# Remove all closing tt tags
	# and also remove the closing div matching the div.alltt
	# Replace div.alltt with pre tag
	# Delete tt tag that is no longer supported in html5
	# Remove all br tags in between pre tags
	# and finally delete the line if it just has whitespace
	sed -e 's!'$cwd'/!!g' \
	    -e 's|replace_with_script|script|g' \
		-e '/<\/TT>/{N;s|<\/TT>||g;/\n<\/DIV>/{s|<\/DIV>|</code></pre>|g}}' \
		-e 's|<DIV CLASS="alltt"[^>]*>|<pre><code>|g' \
		-e 's|<TT>||g' \
		-e '/<pre>/,/<\/pre>/s|<BR>||g' \
		-e '/^\w*$/d' \
	tmp > $f || die "error running sed on $f"
done
