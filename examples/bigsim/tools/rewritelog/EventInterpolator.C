
#include <EventInterpolator.h>

#define DEBUG

using namespace std;

int EventInterpolator::numCoefficients(const string &funcname){
// We create a dummy input stringstream and pass it to parseParameters.
// Then we count how many parameters that function creates
    string temp("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
    istringstream temp2(temp);
    return parseParameters(funcname,temp2,false).second.size();
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

pair<int,vector<double> > EventInterpolator::parseParameters(const string &funcname, istringstream &param_stream, const bool log){
    double temp;
    return parseParameters(funcname,param_stream,temp, log);
}


pair<int,vector<double> > EventInterpolator::parseParameters(const string &funcname, istringstream &line, double &time, const bool log){
    vector<double> params;
    int category=0;

    if( funcname == string("calc_self_energy_merge_fullelect") ||
        funcname == string("calc_pair_energy_merge_fullelect") ||
        funcname == string("calc_self_merge_fullelect") ||
        funcname == string("calc_pair_merge_fullelect") ||
        funcname == string("calc_self") ||
        funcname == string("calc_pair") ||
        funcname == string("calc_self_energy") ||
        funcname == string("calc_pair_energy") ){

        double d1,d2,d3,d4,d5,d6,d7,d8,d9, t1;
        int i1,i2,i3,i4,i5,i6;

        line >> d1 >> d2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >>
                d3 >> d4 >> d5 >> d6 >> d7 >> d8 >> t1;

        if(log)
            log1 << funcname << "\t" << t1 << "\t"  << d1  << "\t" << d2 << "\t"  << d3  << "\t" << d4  << "\t" << d5 << "\t"  << d6  << "\t" << d7  << "\t" << d8 << "\t" << i1  << "\t" << i2 << "\t"  << i3  << "\t" << i4  << "\t" << i5 << "\t"  << i6 << "\t" << distance(i1,i2,i3,i4,i5,i6) << endl;

        params.push_back( 1.0);

//         params.push_back( 1.0 / distance );
        params.push_back( d3 );
        params.push_back( d4 );
        params.push_back( d5 );
		//        params.push_back( d2 );
		//        params.push_back( d3 );
		//        params.push_back( d4 );
        params.push_back( min(d1,d2)*d1*d2 );
        params.push_back( d1*d2 );

        category = distance(i1,i2,i3,i4,i5,i6 );

        time = t1;
    }
    else if(funcname == string("angle") || funcname == string("dihedrals")){
        double d1, d2, t1;
        line >> d1 >> d2 >> t1;

        if(log)
            log1 << funcname << "\t" << t1 << "\t"  << d1  << "\t" << d2 << endl;

        params.push_back( 1.0);
        params.push_back( d2 );
        params.push_back( d2*d2 );

        time = t1;

    }
    else if(funcname == string("*integrate*")){
        double d1, d2, d3, d4, d5, d6, d7, t1;
        line >> d1 >> d2 >> d3 >> d4 >> d5 >> d6 >> d7 >> t1;

        if(log)
            log1 << funcname << "\t" << t1 << "\t" << d1 << "\t" << d2 << "\t" << d3 << "\t" << d4 << "\t" << d5 << "\t" << d6 << "\t" << d7 << endl;

        params.push_back( 1.0);
        params.push_back( d2 );
        params.push_back( d2*d2 );
        time = t1;

    }
    else {
        cerr << "FATAL ERROR: Don't know how to read parameters for function " << funcname << endl;
        throw new runtime_error("unknown function");
    }

    return pair<int,vector<double> >(category,params);
}





EventInterpolator::EventInterpolator(char *table_filename){
    exact_matches=0;
    exact_positive_matches=0;
    approx_matches=0;
    approx_positive_matches=0;

	log1.open("log1");

    cout << "Loading timings file: " << table_filename << endl;
    ifstream accurateTimeTable(table_filename);
    if(! accurateTimeTable.good() ){
        cerr << "FATAL ERROR: Couldn't open file(perhaps it doesn't exist): " << table_filename << endl;
        throw new runtime_error("missing file");
    }


    // First pass, scan through cycle accurate time file to count
    // how many samples there are for each function
    while(accurateTimeTable.good()){
        string line_s;
        getline(accurateTimeTable,line_s);
        istringstream line(line_s);

        string temp("");
        while(temp != string("TRACEBIGSIM") && line.good() && accurateTimeTable.good() ){
            line >> temp;
        }
        line >> temp; // gobble up one more worthless bit of input line

        if(line.good() && accurateTimeTable.good()){
            string funcname;
            line >> funcname;

			double time;
			fullParams params = parseParameters(funcname,line,time,false);
            funcIdentifier func(funcname,params.first);
            sample_count[func]++;
        }
    }

    // Create a gsl interpolator workspace for each event/function
    for(map<funcIdentifier,unsigned long>::iterator i=sample_count.begin(); i!=sample_count.end();++i){
        funcIdentifier func = (*i).first;
        unsigned long samples = (*i).second;
        cout << "     > " << func.first << "," << func.second << " has " << samples << " sampled timings" << endl;

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

    accurateTimeTable.close();
    ifstream accurateTimeTable2(table_filename);

#ifdef WRITESTATS
    ofstream statfile("stats-out");
#endif

    // Second pass, scan through cycle accurate time file
    while(accurateTimeTable2.good()){
        string line_s;
        getline(accurateTimeTable2,line_s);
        istringstream line(line_s);

        string temp("");
        while(temp != string("TRACEBIGSIM") && line.good() && accurateTimeTable2.good() ){
            line >> temp;
        }
        line >> temp; // gobble up one more worthless bit of input line

        if(line.good() && accurateTimeTable2.good()){
            string funcname;
            line >> funcname;

            double time;
            fullParams params = parseParameters(funcname,line,time,false);

            funcIdentifier func(funcname,params.first);

            unsigned i = Xcount[func] ++;
            gsl_matrix * x = X[func];
            accurateTimings[pair<funcIdentifier,vector<double> >(func,params.second)]=time;

#ifdef WRITESTATS
            statfile << funcname << "\t" << time << endl;
#endif

            for(int param_index=0;param_index<params.second.size();++param_index){
                gsl_matrix_set(x,i,param_index, params.second[param_index]);
            }
            gsl_vector_set(y[func],i,time);

        }
    }

#ifdef WRITESTATS
    statfile.close();
#endif

    // Perform a sanity check now

    for(map<funcIdentifier, gsl_multifit_linear_workspace *>::iterator i=work.begin();i!=work.end();++i){
        if(sample_count[(*i).first]!=Xcount[(*i).first]){
          cerr << "FATAL ERROR: sanity check failed: " << sample_count[(*i).first] << "!=" << Xcount[(*i).first] << "  :(" << endl;
       throw new runtime_error("sanity check failed");
        }
    }

    cout << "Performing Least Squared Fit to sampled time data" << endl;

    //  Now do Least Square Fit: Find C where y=Xc
    map<funcIdentifier, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        funcIdentifier func = (*i).first;
        assert(! gsl_multifit_linear(X[func],y[func],c[func],cov[func],&chisqr[func],work[func]));
        gsl_matrix_free(X[func]);
        gsl_vector_free(y[func]);
        cout << "     > " << func.first << "," << func.second << " has chisqr=" << chisqr[func] << endl;
    }



    // Load in Parameter File which maps event id to function name and parameters
    cout << "Loading parameter files" << endl;

    for(int i=0;i<102400;++i){
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
                        fullParams params = parseParameters(funcname,line,false);
                        funcIdentifier func(funcname,params.first);
                        Xcount[func] ++;
                        eventparams[pair<unsigned,unsigned>(i,eventid)] = pair<funcIdentifier,vector<double> >(func,params.second);
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



bool EventInterpolator::haveExactTime(const unsigned pe, const unsigned eventid){
    return haveExactTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}

bool EventInterpolator::haveExactTime(const pair<funcIdentifier,vector<double> > &p) {
    return haveExactTime(p.first,p.second);
}

bool EventInterpolator::haveExactTime(const funcIdentifier& func, const vector<double> &p){
    return (accurateTimings.find(pair<funcIdentifier,vector<double> >(func,p)) != accurateTimings.end());
}

double EventInterpolator::lookupExactTime(const unsigned pe, const unsigned eventid){
    return lookupExactTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}

double EventInterpolator::lookupExactTime(const pair<funcIdentifier,vector<double> > &p) {
    return lookupExactTime(p.first,p.second);
}

double EventInterpolator::lookupExactTime(const funcIdentifier& func, const vector<double> &p){
    double val = accurateTimings[pair<funcIdentifier,vector<double> >(func,p)];
    exact_matches++;
    if(val>=0.0)
        exact_positive_matches++;
	else
	  cout << "exact negative match = " << val << endl;
    return val;
}

/** If we have a parameterfile entry for the requested pe,eventid pair */
double EventInterpolator::haveNewTiming(const unsigned pe, const unsigned eventid) {
    return eventparams.find( pair<unsigned,unsigned>(pe,eventid) ) != eventparams.end();
}

double EventInterpolator::predictTime(const unsigned pe, const unsigned eventid) {
    return predictTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}

double EventInterpolator::predictTime(const pair<funcIdentifier,vector<double> > &p) {
    return predictTime(p.first,p.second);
}

bool EventInterpolator::canInterpolateFunc(const funcIdentifier& func){
    return (work.find(func) != work.end());
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




