#include <EventInterpolator.h>

//#define DEBUG
#define PRETEND_NO_ENERGY

using namespace std;


/** Create a vector such that "count" items out of "length" in the vector are set to true */
vector<bool> random_select(int count, int length){
  vector<bool> v(length);
  int c=0;
  // Initialize all to false
  for(int i=0;i<length;++i)
	v[i] = false;
  while(c < count){
	// find a random entry int the vector
	int r = rand() % length;
	// mark it as true, incrementing c if this is the first time we mark it
	if(v[r] == false){
	  c++;
	  v[r] = true;
	}
  }
  return v;
}

template <typename T, typename T2>
multimap<T,T2> random_select_multimap(multimap<T,T2> input_map , int count){
    assert(input_map.size() >= count);
    vector<bool> which_to_keep = random_select(count, input_map.size());
    multimap<T,T2> output_map;
    int len = which_to_keep.size();
    typename multimap<T, T2>::iterator itr=input_map.begin();
    for(int i=0; i<len; ++i, ++itr){
        if(which_to_keep[i] == true){
          // Do something like this to keep them around:
            output_map.insert(make_pair((*itr).first, (*itr).second));
        }
    }
//     assert(output_map.size()==count);
    return output_map;
}

int distance(int i1, int i2, int i3, int i4, int i5, int i6){
    int x1 =(abs(13+i1-i4)%13)*4;
    int x2 =(abs(13+i4-i1)%13)*4;
    int x = min(x1,x2);

    int y1 =(abs(6+i2-i5)%6)*2;
    int y2 =(abs(6+i5-i2)%6)*2;
    int y = min(y1,y2);

    int z1 =(abs(4+i3-i6)%4)*1;
    int z2 =(abs(4+i6-i3)%4)*1;
    int z = min(z1,z2);

    return x+y+z;

}

string parseStreamUntilSquiggle(istream &in){
  string temp;
  ostringstream out;
  in >> temp;
  while(temp != string("}") && in.good()){

	out << temp << " ";

	in >> temp;
  }

  return out.str();
}


/** Record the number of parameters for the given type of event. Subsequent invocation for the same function identifier will fail if num_params doesn't match the recorded value from the first invocation  */
void EventInterpolator::recordNumCoefficients(const string &f, int c){
    // find existing entry and verify it matches
    if(number_of_coefficients.find(f) != number_of_coefficients.end()){
        if(number_of_coefficients[f] != c){
            cerr << "Number of coefficients for function " << f << " is not consistent" << endl;
            exit(-5);
        }
    } else {
        // create if no entry was found
        number_of_coefficients[f] = c;
    }
}


int EventInterpolator::numCoefficients(const string &funcname){
     if(number_of_coefficients.find(funcname) == number_of_coefficients.end()){
            cerr << "Number of coefficients looked up before it was set." << endl;
            exit(-5);
    }
    int num = number_of_coefficients[funcname];
    return num;
}

counterValues EventInterpolator::parseCounters(istringstream &line){
    vector<long> c;
    long val;
    while(line.good()){
        line >> val;
        if(line.good()){
            c.push_back(val);
        }
    }
    return c;
}




/** First parameter is a category or subclass for a particular timed region. For example, we wish to consider computation for adjacent patches differently than patches that are neighbors of adjacent patches. This allows a single timed region to be broken out and treated as a number of different cases.

@note The use should modify this function to incorporate better models of the interpolation function basis

*/
pair<int,vector<double> > EventInterpolator::parseParameters(const string &funcname, istringstream &line){
    vector<double> params;
    int category=0;

    if( funcname == string("calc_self_energy_merge_fullelect") ||
        funcname == string("calc_pair_energy_merge_fullelect") ||
	funcname == string("calc_self_energy_fullelect") ||
        funcname == string("calc_pair_energy_fullelect") ||
        funcname == string("calc_self_merge_fullelect") ||
        funcname == string("calc_pair_merge_fullelect") ||
        funcname == string("calc_self") ||
        funcname == string("calc_pair") ||
        funcname == string("calc_self_energy") ||
        funcname == string("calc_pair_energy") ){

        double d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14;

        line >> d1 >> d2 >> d3 >> d4 >> d5 >> d6 >> d7 >> d8 >> d9 >> d10 >> d11 >> d12 >> d13 >> d14;

        params.push_back( 1.0);

        params.push_back( d3 );
        params.push_back( d4 );
        params.push_back( d5 );

        params.push_back( min(d1,d2)*d1*d2 );
        params.push_back( d1*d2 );

        category = distance((int)d9,(int)d10,(int)d11,(int)d12,(int)d13,(int)d14);

    }
    else { /** @note default model is quadratic in each variable */
        double d;
        int count=0;
        params.push_back( 1.0);
        while(line.good()){
            line >> d;
            if(line.good()){
                params.push_back(d);
                params.push_back(d*d);
                count++;
            }
        }
    }
    recordNumCoefficients(funcname,params.size());
    return make_pair(category,params);
}


void EventInterpolator::LoadTimingsFile(char *table_filename){

  cout << "Loading timings file: " << table_filename << endl;
  ifstream accurateTimeTable(table_filename);
  if(! accurateTimeTable.good() ){
    cerr << "FATAL ERROR: Couldn't open file(perhaps it doesn't exist): " << table_filename << endl;
    throw new runtime_error("missing file");
  }


  // Scan through cycle accurate time file inserting the entries into a map
  while(accurateTimeTable.good())  {
    string line_s;
    getline(accurateTimeTable,line_s);
    istringstream line(line_s);

    // Expected format is:
    // TRACEBIGSIM: event:{ eid } time:{ t } PAPI:{ list } params:{ list }
    // params must currently come after event and time.

    string funcname(""), paramstring(""), counterstring("");
    double time = -1;
    string temp;
    line >> temp;
    if(temp == string("TRACEBIGSIM:")){
      // parse the next piece of the line
      string field;
      line >> field;
      while(line.good() && accurateTimeTable.good()){
        if(field == string("event:{")){
          line >> funcname;
          line >> temp; // gobble up the '}'
          if(temp != string("}"))
            cerr << "ERROR: malformed entry in expected times file" << endl;
        } else if (field == string("time:{")){
          line >> time;
          line >> temp; // gobble up the '}'
          if(temp != string("}"))
            cerr << "ERROR: malformed entry in expected times file" << endl;
        } else if (field == string("PAPI:{")){
          counterstring = parseStreamUntilSquiggle(line);
        } else if (field == string("params:{")){
          paramstring = parseStreamUntilSquiggle(line);
        } else {
          cout << "Unknown event field: " << field  << endl;
        }
        line >> field;
      }

      if(funcname != string("") && time != -1){

        istringstream paramstream(paramstring);
        fullParams params = parseParameters(funcname,paramstream);


        funcIdentifier func(funcname,params.first);

        sample_count[func]++;

        cycle_accurate_time_sum[funcname] += time;

        accurateTimings.insert(make_pair(make_pair(func,params.second),time));

        istringstream counterstream(counterstring);
        counterValues counters = parseCounters(counterstream);
        papiTimings.insert(make_pair(counters,time));

      } else {
        cerr << "malformed TRACEBIGSIM: line" << endl;
      }

    }

  }


  accurateTimeTable.close();

}

void EventInterpolator::AnalyzeTimings(double sample_rate){

  std::map<string,double>::iterator cycle_i;
  cout << "\nThe following table displays the total cycle count for each event in the \n"
    "cycle-accurate timings file. This may be useful for back of the envelope \n"
    "calculations for expected performance " << endl << endl;
  cout << "\t|===============================|===================|" << endl;
  cout << "\t|                        event  | total time (sec)  |" << endl;
  cout << "\t|-------------------------------|-------------------|" << endl;
  for(cycle_i= cycle_accurate_time_sum.begin();cycle_i!=cycle_accurate_time_sum.end();++cycle_i){
    cout << "\t|";
    assert((*cycle_i).first.length() <= 30); // the event names should be at most 30 characters
    for(int i=0;i<30-(*cycle_i).first.length();++i)
      cout << " ";
    cout << (*cycle_i).first << " | ";
    cout.width(17);
    cout << (*cycle_i).second;
    cout << " |" << endl;
  }
  cout << "\t|===============================|===================|" << endl << endl;



  unsigned long sample_keep = (unsigned long) ceil(sample_rate * accurateTimings.size());
  if (sample_rate < 1.0){
    cout << "Randomly dropping " << (1.0-sample_rate)*100 << "% of the cycle accurate timings" << endl;
    accurateTimings = random_select_multimap(accurateTimings, sample_keep);
  }

  analyzeExactVariances();

  cout << "\nThe following table displays the number of timing samples from \n the cycle-accurate file for each event." << endl << endl;
  cout << "\t|===================================|==========================|" << endl;
  cout << "\t|                            event  | number of timing samples |" << endl;
  cout << "\t|-----------------------------------|--------------------------|" << endl;


  // Create a gsl interpolator workspace for each event/function
  for(map<funcIdentifier,unsigned long>::iterator i=sample_count.begin(); i!=sample_count.end();++i){
    funcIdentifier func = (*i).first;
    unsigned long samples = (*i).second;

    cout << "\t|";

    for(int i=0;i<30-func.first.length();++i)
      cout << " ";
    cout << func.first << ",";
    cout.width(3);
    cout << func.second;

    cout << " | ";
    cout.width(24);
    cout << samples << " |" << endl;


    if(samples < numCoefficients(func.first) ){
      cerr << "FATAL ERROR: Not enough input timing samples for " << func.first << "," << func.second << " which has " << numCoefficients(func.first) << " coefficients" << endl;
      throw new runtime_error("samples < numCoefficients");
    }
    else {
      work[func] = gsl_multifit_linear_alloc(samples,numCoefficients(func.first));
      X[func] = gsl_matrix_alloc (samples,numCoefficients(func.first));
      y[func] = gsl_vector_alloc (samples);
      c[func] = gsl_vector_alloc(numCoefficients(func.first));
      cov[func] = gsl_matrix_alloc(numCoefficients(func.first),numCoefficients(func.first));

      for(int i=0;i<numCoefficients(func.first);++i){
        gsl_vector_set(c[func],i,1.0);
      }

    }
  }
  cout << "\t|===================================|==========================|" << endl << endl;


#ifdef WRITESTATS
  ofstream statfile("stats-out");
#endif

  // Fill in values for matrix X and vector Y which will be fed into the least square fit algorithm
  // The data is currently in  accurateTimings[pair<funcIdentifier,vector<double> >(func,params.second)]=time;

  //iterate through accurateTimings
  map< pair<funcIdentifier,vector<double> >, double >::iterator itr;
  for(itr=accurateTimings.begin();itr!=accurateTimings.end();++itr){
    const double &time = (*itr).second;
    const vector<double> &paramsSecond =(*itr).first.second;
    const funcIdentifier func = (*itr).first.first;

    // lookup the next unused entry in X
    unsigned next = Xcount[func] ++;
    gsl_matrix * x = X[func];

    // copy data into array X and Y
    for(int param_index=0;param_index<paramsSecond.size();++param_index){
      gsl_matrix_set(x,next,param_index, paramsSecond[param_index]);
    }
    gsl_vector_set(y[func],next,time);

  }


#ifdef WRITESTATS
  statfile.close();
#endif

  // Perform a sanity check now
  int count1=0, count2=0;
  for(map<funcIdentifier, gsl_multifit_linear_workspace *>::iterator i=work.begin();i!=work.end();++i){
#if DEBUG
    cout << "sample count vs Xcount (" << (*i).first.first << "): " << sample_count[(*i).first] << "?=" << Xcount[(*i).first] << " " << endl;
#endif
    count1 += sample_count[(*i).first];
    count2 += Xcount[(*i).first] ;
  }
  cout << "Samples from cycle accurate file : " << count1 << ".  Keeping only " << count2 << " of them " << endl;

  cout << "Performing Least Squared Fit to sampled time data" << endl;

  //  Now do Least Square Fit: Find C where y=Xc
  cout << "\nThe following table displays the chi^2 measure for how well the model fits the input times." << endl << endl;
  cout << "\t|===================================|=========|=========|" << endl;
  cout << "\t|                            event  |   chi^2 |     chi |" << endl;
  cout << "\t|-----------------------------------|---------|---------|" << endl;

  cout.setf(ios_base::scientific, ios_base::floatfield);
  cout.precision(1);

  map<funcIdentifier, gsl_multifit_linear_workspace *>::iterator i;
  for(i=work.begin();i!=work.end();++i){
    funcIdentifier func = (*i).first;
    assert(! gsl_multifit_linear(X[func],y[func],c[func],cov[func],&chisqr[func],work[func]));
    gsl_matrix_free(X[func]);
    gsl_vector_free(y[func]);

    cout << "\t|";

    for(int i=0;i<30-func.first.length();++i)
      cout << " ";
    cout << func.first << ",";
    cout.width(3);
    cout << func.second;

    cout << " | ";
    cout.width(7);
    cout << chisqr[func];

    cout << " | ";
    cout.width(7);
    cout << sqrt(chisqr[func]) << " |" << endl;

  }
  cout << "\t|===================================|=========|=========|" << endl << endl;

}

/** 
	The model is currently a linear combination of the PAPI counters.
	The first coefficient is associated with a constant value 1.0, and all 
	other coefficients are associated with the performance counters in order.
*/
void EventInterpolator::AnalyzeTimings_PAPI(){

    cout << "Analyzing the PAPI performance counters. Using all samples" << endl;

    gsl_multifit_linear_workspace * work;
    gsl_vector * c;
    gsl_matrix * cov;
    double chisqr;

    gsl_matrix * X;  // Each row of matrix is a set of parameters  [1, a, b, c, ... ] for each input parameter set
    gsl_vector *y;  // vector of cycle accurate times for each input parameter setAnalyzeTimings(

    if(papiTimings.begin() != papiTimings.end()){
        int numCounters = (*papiTimings.begin()).first.size();
        int numCoefficients = numCounters + 1; // currently we use a linear function of the counter values
#define MANUAL_COUNTER_SELECTION
#ifdef MANUAL_COUNTER_SELECTION	
	numCoefficients = 2;
#endif
        int samples = papiTimings.size();

        // Create a gsl interpolator workspace, and populate with values from papiTimings
        work = gsl_multifit_linear_alloc(samples,numCoefficients);
        X = gsl_matrix_alloc (samples,numCoefficients);
        y = gsl_vector_alloc (samples);
        c = gsl_vector_alloc(numCoefficients);
        cov = gsl_matrix_alloc(numCoefficients,numCoefficients);

        // Initialize c. Probably unneeded.
        for(int i=0;i<numCoefficients;++i){
            gsl_vector_set(c,i,1.0);
        }


        // Build matrix X and vector y from the samples
        int whichSample=0;
        for(map<counterValues,double>::iterator itr=papiTimings.begin(); itr!=papiTimings.end();++itr){
            const double &time = (*itr).second;
            const vector<long> &counterValues =(*itr).first;

#ifndef MANUAL_COUNTER_SELECTION
            // put a constant coefficient in there
            gsl_matrix_set(X, whichSample, 0, 1.0);

            // For each PAPI counter
            assert(counterValues.size() == numCounters);
            for(int counter_index=0;counter_index<counterValues.size();++counter_index){
                gsl_matrix_set(X, whichSample, counter_index+1, (double)counterValues[counter_index]);
            }
#else 

 	    gsl_matrix_set(X, whichSample, 0, 1.0);
 	    gsl_matrix_set(X, whichSample, 1, (double)counterValues[1]);

#endif



            gsl_vector_set(y, whichSample, (double)time);
            whichSample++;
        }


        //  Now do Least Square Fit: Find C where y=Xc
        assert(! gsl_multifit_linear(X,y,c,cov,&chisqr,work));
        gsl_matrix_free(X);
        gsl_vector_free(y);

        cout << "Fit data to PAPI based model to get the following results:" << endl;
        cout << "    > chisqr=" << chisqr << endl;
        cout << "    > coefficients=";
        for(int i=0;i<numCoefficients;i++){
            cout.setf(ios_base::scientific, ios_base::floatfield);
            cout.precision(2);
            cout << gsl_vector_get(c, i) <<  " ";
        }
	cout << endl;

    }
    else {
        cout << "No PAPI timings found in the file" << endl;
    }

}


void EventInterpolator::LoadParameterFiles(){

// Load in Parameter File which maps event id to function name and parameters
  cout << "Loading parameter files (May take a while)" << endl;

  for(int i=0;true;++i){
    char name[512];
    sprintf(name,"param.%d",i);
    ifstream parameterEventTable(name);

    if(parameterEventTable.good()){
#ifdef DEBUG
      cout << "     >  Loading " << name << endl;
#endif

      while(parameterEventTable.good()){
        string line_s;
        getline(parameterEventTable,line_s);
        istringstream line(line_s);

        if(parameterEventTable.good()){
          string t1, t2;
          string funcname;
          unsigned eventid;

          line >> eventid;
          line >> t1 >> t2;
          line >> funcname;

          if(t1 == string("TRACEBIGSIM")){
#ifdef PRETEND_NO_ENERGY
            if(funcname == string("calc_pair_energy")){\
              funcname = "calc_pair";
            }
            else if(funcname == string("calc_self_energy")){
              funcname = "calc_self";
            }
#endif

            fullParams params = parseParameters(funcname,line);
            funcIdentifier func(funcname,params.first);
            Xcount[func] ++;
            eventparams[make_pair(i,eventid)] = make_pair(func,params.second);
          }
        }
      }

    }
    else{ // file was no good
      break;
    }

    parameterEventTable.close();
  }
}


EventInterpolator::EventInterpolator(char *table_filename, double sample_rate) :
  exact_matches(0),
  exact_positive_matches(0),
  approx_matches(0),
  approx_positive_matches(0)
{
    LoadTimingsFile(table_filename);
    AnalyzeTimings(sample_rate);
   // AnalyzeTimings_PAPI();
    LoadParameterFiles();
}


bool EventInterpolator::haveExactTime(const unsigned pe, const unsigned eventid){
    return haveExactTime(eventparams[make_pair(pe,eventid)]);
}

bool EventInterpolator::haveExactTime(const pair<funcIdentifier,vector<double> > &p) {
    return haveExactTime(p.first,p.second);
}

bool EventInterpolator::haveExactTime(const funcIdentifier& func, const vector<double> &p){
  return (accurateTimings.find(make_pair(func,p)) != accurateTimings.end());
}

double EventInterpolator::lookupExactTime(const unsigned pe, const unsigned eventid){
    return lookupExactTime(eventparams[make_pair(pe,eventid)]);
}

double EventInterpolator::lookupExactTime(const pair<funcIdentifier,vector<double> > &p) {
    return lookupExactTime(p.first,p.second);
}

double EventInterpolator::lookupExactTime(const funcIdentifier& func, const vector<double> &p){
    const  pair<funcIdentifier,vector<double> > key(func,p);
    const int count = accurateTimings.count(key);

	pair<timings_type::iterator, timings_type::iterator> itrs = accurateTimings.equal_range(key);
	double time_sum=0;
	for(timings_type::iterator itr=itrs.first; itr!=itrs.second; ++itr){
	  time_sum += (*itr).second;
	}
	double time_average = time_sum / count;

    exact_matches++;
	assert(time_average >= 0.0 );
    return time_average;
}


/** Print out some statistics about the timings for any parameter that has multiple associated cycle-accurate timings */
void EventInterpolator::analyzeExactVariances(){

    map< pair<funcIdentifier,vector<double> >, double > variances;
    map< pair<funcIdentifier,vector<double> >, double > means;
	map< funcIdentifier, double > max_std_dev;
	map< funcIdentifier, double > associated_mean;

	// Iterate through items in accurateTimings and store variances
	timings_type::iterator itr;
	for(itr=accurateTimings.begin();itr!=accurateTimings.end();++itr){
	  variances[(*itr).first] = lookupExactVariance( (*itr).first );
	  means[(*itr).first] = lookupExactMean( (*itr).first );
	}

	// Display the variances
	map< pair<funcIdentifier,vector<double> >, double >::iterator vItr;
	for(vItr = variances.begin(); vItr!=variances.end(); ++vItr){
	  double var = (*vItr).second;
	  double stddev = sqrt(var);
	  double mean = means[(*vItr).first];

	  if(var > 0.0){
		if(stddev > max_std_dev[(*vItr).first.first]){
		  max_std_dev[(*vItr).first.first] = stddev;
		  associated_mean[(*vItr).first.first] = mean;
		}
	  }
	}

	// Display the max std dev for any given event
	cout << "\nThe following events have differing exact times for one or more parameter set.\n"
	        "Each line corresponds to whichever parameter list produced the greatest variance\n"
	        "(and equivalently std dev). The mean is the mean timing value associated with the\n"
            " same parameter list of greatest variance" << endl << endl;


	cout << "\t|===================================|=================|=================|==============|" << endl;
	cout << "\t|                            event  |     max std dev |      mean (sec) | sd % of mean |" << endl;
	cout << "\t|-----------------------------------|-----------------|-----------------|--------------|" << endl;

//	cout.setf(ios_base::fixed, ios_base::floatfield);
    cout.setf(ios_base::scientific, ios_base::floatfield);
	cout.precision(3);

	int func_name_field_width=30;
	map< funcIdentifier, double >::iterator sItr;
	for(sItr=max_std_dev.begin(); sItr!=max_std_dev.end(); ++sItr) {
	  double stddev = (*sItr).second;
	  double mean = associated_mean[(*sItr).first];
	  cout << "\t|";
	  for(int i=0;i<func_name_field_width-(*sItr).first.first.length();++i)
		cout << " ";
	  cout << (*sItr).first.first ;
	  cout << ",";
	  cout.width(3);
	  cout << (*sItr).first.second;
	  cout << " | ";
	  cout.width(15);
      cout.setf(ios_base::scientific, ios_base::floatfield);
      cout.precision(3);
	  cout <<  stddev;
	  cout << " | ";
      cout.width(15);
      cout.setf(ios_base::scientific, ios_base::floatfield);
      cout.precision(3);
	  cout << mean << " | ";
	  cout.width(9);
      cout.setf(ios_base::fixed, ios_base::floatfield);
      cout.precision(1);
	  cout << (stddev * 100.0 / mean) << "    | " << endl;
	}
	cout.setf(ios_base::fmtflags(0), ios_base::floatfield);
	cout << "\t|===================================|=================|=================|==============|" << endl << endl;

}

/** Lookup the average timing for a given event and parameter list */
double EventInterpolator::lookupExactVariance(const unsigned pe, const unsigned eventid){
    return lookupExactVariance(eventparams[make_pair(pe,eventid)]);
}

/** Lookup the variance in the timings for a given event and parameter list */
double EventInterpolator::lookupExactVariance(const pair<funcIdentifier,vector<double> > &p) {
    return lookupExactVariance(p.first,p.second);
}

/** Lookup the variance in the timings for a given event and parameter list */
double EventInterpolator::lookupExactVariance(const funcIdentifier& func, const vector<double> &p){
    const  pair<funcIdentifier,vector<double> > key(func,p);
    const int count = accurateTimings.count(key);

	// Compute mean
	pair<timings_type::iterator, timings_type::iterator> itrs = accurateTimings.equal_range(key);
	double time_sum=0;
	for(timings_type::iterator itr=itrs.first; itr!=itrs.second; ++itr){
	  time_sum += (*itr).second;
	}
	const double time_average = time_sum / count;

	// Compute Variance
	double V_sum=0;
	for(timings_type::iterator itr=itrs.first; itr!=itrs.second; ++itr){
	  const double d = (*itr).second-time_average;
	  V_sum += d*d;
	}
	const double Var = V_sum / count;

    return Var;
}

/** Lookup the average timing for a given event and parameter list */
double EventInterpolator::lookupExactMean(const unsigned pe, const unsigned eventid){
    return lookupExactMean(eventparams[make_pair(pe,eventid)]);
}
/** Lookup the average timing for a given event and parameter list */
double EventInterpolator::lookupExactMean(const pair<funcIdentifier,vector<double> > &p) {
    return lookupExactMean(p.first,p.second);
}
/** Lookup the average timing for a given event and parameter list */
double EventInterpolator::lookupExactMean(const funcIdentifier& func, const vector<double> &p){
    const  pair<funcIdentifier,vector<double> > key(func,p);
    const int count = accurateTimings.count(key);

	// Compute mean
	pair<timings_type::iterator, timings_type::iterator> itrs = accurateTimings.equal_range(key);
	double time_sum=0;
	for(timings_type::iterator itr=itrs.first; itr!=itrs.second; ++itr){
	  time_sum += (*itr).second;
	}
	return time_sum / count;
}


/** If we have a parameterfile entry for the requested pe,eventid pair */
double EventInterpolator::haveNewTiming(const unsigned pe, const unsigned eventid) {
    return eventparams.find( make_pair(pe,eventid) ) != eventparams.end();
}

double EventInterpolator::predictTime(const unsigned pe, const unsigned eventid) {
    return predictTime(eventparams[make_pair(pe,eventid)]);
}

double EventInterpolator::predictTime(const pair<funcIdentifier,vector<double> > &p) {
    return predictTime(p.first,p.second);
}

bool EventInterpolator::canInterpolateFunc(const funcIdentifier& func){
    return (work.find(func) != work.end());
}


double EventInterpolator::getNewTiming(const unsigned pe, const unsigned eventid){
  if(haveExactTime(pe,eventid) )
	return lookupExactTime(pe,eventid);
  else
	return predictTime(pe,eventid);
}

double EventInterpolator::predictTime(const funcIdentifier &func, const vector<double> &params) {

    // check name
    if(!canInterpolateFunc(func)){
        cerr << "FATAL ERROR: function name not found in cycle accurate timing file: " << func.first << "," << func.second << endl;
       throw new runtime_error("function name not found");
    }

    // Estimate time for a given set of parameters p
    gsl_vector *desired_params;

    desired_params = gsl_vector_alloc(numCoefficients(func.first));
    assert(numCoefficients(func.first)==params.size());

    for(int i=0;i<params.size();++i){
        gsl_vector_set(desired_params,i,params[i]);
    }

    double desired_time, desired_time_err;
    assert(c[func]);
    assert(cov[func]);
    assert(! gsl_multifit_linear_est(desired_params,c[func],cov[func],&desired_time,&desired_time_err));

    gsl_vector_free(desired_params);


    if(min_interpolated_time.find(func) == min_interpolated_time.end())
        min_interpolated_time[func] = desired_time;
    else
        min_interpolated_time[func] = min( min_interpolated_time[func], desired_time);

    if(max_interpolated_time.find(func) == max_interpolated_time.end())
        max_interpolated_time[func] = desired_time;
    else
        max_interpolated_time[func] = max( max_interpolated_time[func], desired_time);


    approx_matches++;
    if(desired_time>=0.0)
        approx_positive_matches++;

//    gsl_vector_free(desired_params);


    return desired_time;
}


void EventInterpolator::printMinInterpolatedTimes(){

    cout << "The following functions were interpolated(as opposed to approximated:" << endl;

    for(map<funcIdentifier,double>::iterator i=min_interpolated_time.begin();i!=min_interpolated_time.end();++i){
        cout << "   > min predicted/interpolated time for function " << (*i).first.first << "," << (*i).first.second << " is " << (*i).second << " cycles " << endl;
    }
    for(map<funcIdentifier,double>::iterator i=max_interpolated_time.begin();i!=max_interpolated_time.end();++i){
        cout << "   > max predicted/interpolated time for function " << (*i).first.first << "," << (*i).first.second << " is " << (*i).second << " cycles " << endl;
    }

    cout << endl;
}

void EventInterpolator::printMatches(){
    cout << "    > Exact lookup = " << exact_matches << " (" << exact_positive_matches << " positive)" << endl;
    cout << "    > Approximated = " << approx_matches << " (" << approx_positive_matches << " positive)" << endl;
    cout << "    > Total        = " << approx_matches+exact_matches <<  " (" << exact_positive_matches+approx_positive_matches << " positive)" << endl;
}

void EventInterpolator::printCoefficients(){

    for(map<funcIdentifier,gsl_vector*>::iterator i=c.begin();i!=c.end();++i){
        cout << "    > Coefficients for function " << (*i).first.first << "," << (*i).first.second << " :" << endl;
        for(int j=0; j < ((*i).second)->size; ++j){
            cout << "    >    " << j << " is " << gsl_vector_get ((*i).second, j) << endl;
        }
    }

}




/** Free the gsl arrays and workspaces */
EventInterpolator::~EventInterpolator(){
    map<funcIdentifier, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        const funcIdentifier &func = (*i).first;
        gsl_multifit_linear_free(work[func]);
        gsl_matrix_free(cov[func]);
        gsl_vector_free(c[func]);
    }
	log1.close();
}




