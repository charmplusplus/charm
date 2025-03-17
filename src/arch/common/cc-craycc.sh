#This is deliberately vacuous.

#conv-mach.sh already does the right thing. This is just a placeholder so you can choose craycc 

#NOTE: craycc is -O2 by default and is otherwise fairly aggressive. 
#      Further tinkering should be done only as necessary and tested thoroughly.  

#verify that cc is actually the cray compiler, otherwise bail
CRAYCC_test=`cc -v 2>&1 >/dev/null | grep -i 'version' | awk -F' ' '{print $1; exit}'`
test -z "$CRAYCC_test" && echo "cc is not CrayCC, check your modules!" && exit 1
test "$CRAYCC_test" != "Cray"  && echo "cc is not CrayCC, check your modules!" && exit 1
