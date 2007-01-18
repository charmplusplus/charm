
#include <EventInterpolator.h>


using namespace std;

int EventInterpolator::numCoefficients(const string &funcname){
// We create a dummy input stringstream and pass it to readParameters.
// Then we count how many parameters that function creates

string temp("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
istringstream temp2(temp);

return readParameters(funcname,temp2).size();

}

vector<double> EventInterpolator::readParameters(const string &funcname, istringstream &param_stream){
    double temp;
    return readParameters(funcname,param_stream,temp);
}

vector<double> EventInterpolator::readParameters(const string &funcname, istringstream &line, double &time){
    vector<double> params;

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

        double distance = (i1-i4)*(i1-i4) + (i2-i5)*(i2-i5) + (i3-i6)*(i3-i6);

        params.push_back( 1.0);
//         params.push_back( min(d1,d2) );
        params.push_back( d1*d2 );
//         params.push_back( 1.0 / distance );
        params.push_back( d1 );
        params.push_back( d2 );
        params.push_back( d3 );
        params.push_back( d4 );

        time = t1;
    }
    else if(funcname == string("angle") || funcname == string("dihedrals")){
        double d1, d2, t1;
        line >> d1 >> d2 >> t1;

        params.push_back( 1.0);
        params.push_back( d1 );
        params.push_back( d2 );
        time = t1;

    }
    else if(funcname == string("*integrate*")){
        double d1, d2, d3, d4, d5, d6, d7, t1;
        line >> d1 >> d2 >> d3 >> d4 >> d5 >> d6 >> d7 >> t1;

        params.push_back( 1.0);
        params.push_back( d2 );
        time = t1;

    }
    else {
        cerr << "FATAL ERROR: Don't know how to read parameters for function " << funcname << endl;
        throw new runtime_error("unknown function");
    }

    return params;
}





EventInterpolator::EventInterpolator(char *table_filename){
    exact_matches=0;
    exact_positive_matches=0;
    approx_matches=0;
    approx_positive_matches=0;

    cout << "Loading timings file: " << table_filename << endl;
    ifstream accurateTimeTable(table_filename);

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
            sample_count[funcname]++;
            }
        }

    // Create a gsl interpolator workspace for each event/function
    for(map<string,unsigned long>::iterator i=sample_count.begin(); i!=sample_count.end();++i){
        string name = (*i).first;
        unsigned long samples = (*i).second;
        cout << "     > " << name << " has " << samples << " sampled timings" << endl;

        if(samples < numCoefficients(name) ){
            cerr << "FATAL ERROR: Not enough input timing samples for " << name << " which has " << numCoefficients(name) << " coefficients" << endl;
            throw new runtime_error("samples < numCoefficients");
        }
        else {
            work[name] = gsl_multifit_linear_alloc(samples,numCoefficients(name));
            X[name] = gsl_matrix_alloc (samples,numCoefficients(name));
            y[name] = gsl_vector_alloc (samples);
            c[name] = gsl_vector_alloc(numCoefficients(name));
            cov[name] = gsl_matrix_alloc(numCoefficients(name),numCoefficients(name));

            for(int i=0;i<numCoefficients(name);++i){
                gsl_vector_set(c[name],i,1.0);
            }

        }
    }

    accurateTimeTable.close();
    ifstream accurateTimeTable2(table_filename);
    ofstream statfile("stats-out");
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

            unsigned i = Xcount[funcname] ++;
            gsl_matrix * x = X[funcname];

            double time;
            vector<double>params = readParameters(funcname,line,time);

            accurateTimings[pair<string,vector<double> >(funcname,params)]=time;

            statfile << funcname << "\t" << time << endl;


            for(int param_index=0;param_index<params.size();++param_index){
                gsl_matrix_set(x,i,param_index, params[param_index]);
            }
            gsl_vector_set(y[funcname],i,time);

        }
    }
    statfile.close();

    // Perform a sanity check now

    for(map<string, gsl_multifit_linear_workspace *>::iterator i=work.begin();i!=work.end();++i){
        if(sample_count[(*i).first]!=Xcount[(*i).first]){
          cerr << "FATAL ERROR: sanity check failed: " << sample_count[(*i).first] << "!=" << Xcount[(*i).first] << "  :(" << endl;
       throw new runtime_error("sanity check failed");
        }
    }

    cout << "Performing Least Squared Fit to sampled time data" << endl;

    //  Now do Least Square Fit: Find C where y=Xc
    map<string, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        string name = (*i).first;
        assert(! gsl_multifit_linear(X[name],y[name],c[name],cov[name],&chisqr[name],work[name]));
        gsl_matrix_free(X[name]);
        gsl_vector_free(y[name]);
        cout << "     > " << name << " has chisqr=" << chisqr[name] << endl;
    }



    // Load in Parameter File which maps event id to function name and parameters
    cout << "Loading parameter files" << endl;

    for(int i=0;i<102400;++i){
        char name[512];
        sprintf(name,"param.%d",i);
        ifstream parameterEventTable(name);

        if(parameterEventTable.good()){
//             cout << "     >  Loading " << name << endl;

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
                        Xcount[funcname] ++;
                        vector<double>params = readParameters(funcname,line);
                        eventparams[pair<unsigned,unsigned>(i,eventid)] = pair<string,vector<double> >(funcname,params);
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

bool EventInterpolator::haveExactTime(const pair<string,vector<double> > &p) {
    return haveExactTime(p.first,p.second);
}

bool EventInterpolator::haveExactTime(const string& name, const vector<double> &p){
    return (accurateTimings.find(pair<string,vector<double> >(name,p)) != accurateTimings.end());
}

double EventInterpolator::lookupExactTime(const unsigned pe, const unsigned eventid){
    return lookupExactTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}

double EventInterpolator::lookupExactTime(const pair<string,vector<double> > &p) {
    return lookupExactTime(p.first,p.second);
}

double EventInterpolator::lookupExactTime(const string& name, const vector<double> &p){
    double val = accurateTimings[pair<string,vector<double> >(name,p)];
    exact_matches++;
    if(val>0.0)
        exact_positive_matches++;
    return val;
}

/** If we have a parameterfile entry for the requested pe,eventid pair */
double EventInterpolator::haveNewTiming(const unsigned pe, const unsigned eventid) {
    return eventparams.find( pair<unsigned,unsigned>(pe,eventid) ) != eventparams.end();
}

double EventInterpolator::predictTime(const unsigned pe, const unsigned eventid) {
    return predictTime(eventparams[pair<unsigned,unsigned>(pe,eventid)]);
}

double EventInterpolator::predictTime(const pair<string,vector<double> > &p) {
    return predictTime(p.first,p.second);
}

bool EventInterpolator::canInterpolateName(const string& name){
    return (work.find(name) != work.end());
}

double EventInterpolator::predictTime(const string &name, const vector<double> &params) {

    // check name
    if(!canInterpolateName(name)){
        cerr << "FATAL ERROR: function name not found in cycle accurate timing file: " << name << endl;
       throw new runtime_error("function name not found");
    }

    // Estimate time for a given set of parameters p
    gsl_vector *desired_params;

    desired_params = gsl_vector_alloc(numCoefficients(name));
    assert(numCoefficients(name)==params.size());

    for(int i=0;i<params.size();++i){
        gsl_vector_set(desired_params,i,params[i]);
    }

    double desired_time, desired_time_err;
    assert(c[name]);
    assert(cov[name]);
    assert(! gsl_multifit_linear_est(desired_params,c[name],cov[name],&desired_time,&desired_time_err));

    gsl_vector_free(desired_params);


    if(min_interpolated_time.find(name) == min_interpolated_time.end())
        min_interpolated_time[name] = desired_time;
    else
        min_interpolated_time[name] = min( min_interpolated_time[name], desired_time);

    if(max_interpolated_time.find(name) == max_interpolated_time.end())
        max_interpolated_time[name] = desired_time;
    else
        max_interpolated_time[name] = max( max_interpolated_time[name], desired_time);


    approx_matches++;
    if(desired_time>0.0)
        approx_positive_matches++;

    return desired_time;
}


void EventInterpolator::printMinInterpolatedTimes(){
    for(map<string,double>::iterator i=min_interpolated_time.begin();i!=min_interpolated_time.end();++i){
        cout << "   > min interpolated time for function " << (*i).first << " is " << (*i).second << " cycles " << endl;
    }
    for(map<string,double>::iterator i=max_interpolated_time.begin();i!=max_interpolated_time.end();++i){
        cout << "   > max interpolated time for function " << (*i).first << " is " << (*i).second << " cycles " << endl;
    }
}

void EventInterpolator::printMatches(){
    cout << " Exact  Matches = " << exact_matches << " (" << exact_positive_matches << " positive)" << endl;
    cout << " Approx Matches = " << approx_matches << " (" << approx_positive_matches << " positive)" << endl;
}

void EventInterpolator::printCoefficients(){

    for(map<string,gsl_vector*>::iterator i=c.begin();i!=c.end();++i){
        cout << "    > Coefficients for function " << (*i).first << " :" << endl;
        for(int j=0; j < ((*i).second)->size; ++j){
            cout << "    >    " << j << " is " << gsl_vector_get ((*i).second, j) << endl;
        }
    }

}




/** Free the gsl arrays and workspaces */
EventInterpolator::~EventInterpolator(){
    map<string, gsl_multifit_linear_workspace *>::iterator i;
    for(i=work.begin();i!=work.end();++i){
        string name = (*i).first;
        gsl_multifit_linear_free(work[name]);
        gsl_matrix_free(cov[name]);
        gsl_vector_free(c[name]);
    }
}




