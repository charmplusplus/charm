JITRANS=../jitrans/jitrans

CLASSFILES=Proxy_Main.class \
           Proxy_RingNode.class \
           Main.class \
           RingNode.class \
           RingMessage.class \
           RegisterAll.class 

all:$(CLASSFILES)

Proxy_Main.java: Main.ji
	$(JITRANS) Main

Proxy_RingNode.java: RingNode.ji
	$(JITRANS) RingNode

RegisterAll.java: Main.ji RingNode.ji RingMessage.java
	$(JITRANS) -register -classes Main RingNode -messages RingMessage

RegisterAll.class: RegisterAll.java
	javac RegisterAll.java

Proxy_Main.class: Proxy_Main.java
	javac Proxy_Main.java

Proxy_RingNode.class: Proxy_RingNode.java
	javac Proxy_RingNode.java

RingMessage.class: RingMessage.java
	javac RingMessage.java

Main.class: Main.java
	javac Main.java

RingNode.class: RingNode.java
	javac RingNode.java

clean:
	rm -f $(CLASSFILES) Proxy_*.java RegisterAll.java
