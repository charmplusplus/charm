
all: small run

t:
	echo are you on will?
	w
	g++ -O3 -o mm2mkn mm2mkn.C
	/usr/bin/time ./mm2mkn


max:
	# dont do -O3, the loop is compiled away
	g++ -o maxflops maxflops.c
	# 20 Gig float ops takes:
	/usr/bin/time ./maxflops
	# prowess 1.5 GF

small:
	g++ -O3 -o mm1 mm1.C
	g++ -O3 -o mm1mkn mm1mkn.C
	g++ -O3 -o mm2 mm2.C
	g++ -O3 -o mm2mkn mm2mkn.C

run:
	/usr/bin/time ./mm1mkn
	/usr/bin/time ./mm2mkn
	/usr/bin/time ./mm1
	/usr/bin/time ./mm2

big:
	g++ -DBIG -O3 -o mm1 mm1.C
	g++ -DBIG -O3 -o mm1mkn mm1mkn.C
	g++ -DBIG -O3 -o mm2 mm2.C
	g++ -DBIG -O3 -o mm2mkn mm2mkn.C
	/usr/bin/time ./mm2mkn
	/usr/bin/time ./mm2

clean:
	rm mm1 mm1mkn mm2 mm2mkn
