


cp_effects : cp_effects.cpp cp_effects.h
	 ../../../bin/charmc  cp_effects.cpp -o cp_effects

cp_effects.h cp_effects.cpp : cp_effects.pl cp_effects.txt
	./cp_effects.pl

clean : 
	rm -rf cp_effects *~