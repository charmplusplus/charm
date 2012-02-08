class someData 
{
 public:
 someData(int _howBig):howBig(_howBig){data=new int[howBig];}
  someData(){data=NULL; howBig=0;}
  void pup(PUP::er &p)
    {
      // remember to pup your superclass if there is one
      p|howBig;
      if(p.isUnpacking())
	data=new int[howBig];
      PUParray(p,data,howBig);
    }

  inline someData &operator=(const someData &indata) {
    if(data && howBig>0) delete [] data;
    howBig=indata.howBig;
    data=new int[howBig];
    for(int i=0; i<howBig; ++i) data[i]=indata.data[i];
    return *this;
  }

  
  ~someData(){if (data); delete [] data;}
  int howBig;
  int *data;
};
