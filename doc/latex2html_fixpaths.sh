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

[ -x manual ] || die "fixpaths.sh requires a manual/ directory"

cp -r fig figs manual/

for f in `echo manual/*.html`
do
	echo "Converting $f"
	cwd=`pwd`
	cwd=`echo $cwd | sed -e 's@/home/net@/expand/home@'`
	sed -e 's!'`pwd`'/!!g' $f > tmp || die "error running sed on $f"
	mv $f $f.bak || die "error backing up $f"
	sed -e 's!'$cwd'/!!g' tmp > $f || die "error running sed on $f"
	#mv tmp $f || die "error replacing $f"
done
