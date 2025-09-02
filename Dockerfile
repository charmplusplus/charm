FROM mpioperator/openmpi

RUN apt update && apt install -y build-essential zlib1g-dev ca-certificates cmake git

RUN apt update \
    && apt install -y --no-install-recommends \
        g++ \
        gfortran \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/charmplusplus/charm.git
RUN mkdir /home/mpiuser/charm
COPY . /home/mpiuser/charm
RUN cd charm && git checkout shrinkexpand-mpi && ./build charm++ mpi-linux-x86_64 --enable-shrinkexpand -j8 --force --with-production

RUN cd charm/examples/charm++/shrink_expand && make clean && make
RUN cd charm/examples/charm++/shrink_expand/jacobi2d-iter && make clean && make
RUN cd charm/examples/charm++/shrink_expand/startup && make clean && make
RUN mkdir /app
RUN cp charm/examples/charm++/shrink_expand/jacobi2d-iter/charmrun /app/
RUN cp charm/examples/charm++/shrink_expand/jacobi2d-iter/charmrun_elastic /app/
RUN cp charm/examples/charm++/shrink_expand/jacobi2d-iter/jacobi2d /app/
RUN cp charm/examples/charm++/shrink_expand/startup/startup /app/
RUN chmod 777 /app
