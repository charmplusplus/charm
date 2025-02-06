#include "streamtest.decl.h"
#include "./includes/utils.hpp"
#include <iostream>
#include <time.h>
#include <json.hpp>
#include <sstream>

#define NUM_READERS 1
#define NUM_VALIDATORS 4
#define NUM_FILTERS 4 
#define NUM_TRANSFORMERS 4
#define NUM_WRITERS 1
// #define 

using json = nlohmann::json;


class Main : public CBase_Main {
	CProxy_Readers readers;
	CProxy_Filters filters;
	CProxy_Transformers transformers;
	CProxy_Validators validators;
	CProxy_Writers writers;
	int count = 0;

public:
	Main_SDAG_CODE
	Main(CkArgMsg* m){
		int num_records = atoi(m->argv[1]);
		// mainProxy = thisProxy;
		json records = generate_and_save_json(num_records, "generated_inputs.json");
		delete m;
		
		readers = CProxy_Readers::ckNew(num_records, records.dump(), thisProxy, NUM_READERS);
		
		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::readersValidatorsStreamMade(0), thisProxy));
	}
	Main(CkMigrateMessage* msg) {}
	void pup(PUP::er &p) {}
	void readersValidatorsStreamMade(Ck::Stream::StreamIdMessage* msg) {
		size_t id = msg -> id;
		CkPrintf("Got the stream id: %d\n", id);
		
		validators = CProxy_Validators::ckNew(thisProxy, NUM_VALIDATORS);
		readers.setOutputStreamId(id);
		validators.setInputStreamId(id);
		// Ck::Stream::createNewStream(CkCallback(CkIndex_Main::validatorsFiltersStreamMade(0), thisProxy));
	};

	void validatorsFiltersStreamMade(Ck::Stream::StreamIdMessage* msg) {
		size_t id = msg -> id;
		CkPrintf("Got the stream id: %d\n", id);
		filters = CProxy_Filters::ckNew(NUM_FILTERS);
		validators.setOutputStreamId(id);
		filters.setInputStreamId(id);
		
		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::filtersTransformersStreamMade(0), thisProxy));
	}
	void filtersTransformersStreamMade(Ck::Stream::StreamIdMessage* msg) {
		size_t id = msg -> id;
		CkPrintf("Got the stream id: %d\n", id);
		transformers = CProxy_Transformers::ckNew(NUM_TRANSFORMERS);
		filters.setOutputStreamId(id);
		transformers.setInputStreamId(id);

		Ck::Stream::createNewStream(CkCallback(CkIndex_Main::transformersWritersStreamMade(0), thisProxy));
	}
	void transformersWritersStreamMade(Ck::Stream::StreamIdMessage* msg) {
		size_t id = msg -> id;
		CkPrintf("Got the stream id: %d\n", id);
		writers = CProxy_Writers::ckNew(NUM_WRITERS);
		transformers.setOutputStreamId(id);
		writers.setInputStreamId(id);
	}

	void allComplete() {
		++count;
		if (count == 10) {
			CkPrintf("All stages done, exiting!\n");
			CkExit(0);	
		}
	}
};

class Readers : public CBase_Readers {
	StreamToken output_id = -1; 
	std::string records_str = "";
	int num_records;
	CProxy_Main mainProxy;
public:
	Readers_SDAG_CODE
	Readers() {}
	Readers(int num_records, std::string records_str, CProxy_Main mainProxy): num_records(num_records), records_str(records_str), mainProxy(mainProxy) {}
	void pup(PUP::er &p) {}

	void setOutputStreamId(StreamToken id) {
		output_id = id;
		beginWork();
	}
	Readers(CkMigrateMessage* msg) {};


	void beginWork() {
		if (output_id == -1) return;
		CkPrintf("Actually beginning work for Readers\n");
		json records = json::parse(records_str);
		for (auto& item : records) {
			std::string json_string = item.dump();
			CkPrintf("Readers putting to %d,%d, %s\n", output_id,json_string.size() + 1, json_string.c_str());

			Ck::Stream::putRecord(output_id, (void*)json_string.c_str(), sizeof(char) * json_string.size() + 1);
			CkPrintf("WE HAVE PUT\n");
			Ck::Stream::flushLocalStream(output_id);
		}
		Ck::Stream::closeWriteStream(output_id);
	}
};

class Validators : public CBase_Validators {
	StreamToken input_id = -1;	// stream to fetch input from
	StreamToken output_id = -1; // stream to output to 
	CProxy_Main mainProxy;


public:
	Validators_SDAG_CODE
	Validators(CProxy_Main mainProxy): mainProxy(mainProxy) {}
	Validators(CkMigrateMessage* msg) {};
	void pup(PUP::er &p) {}

	void setInputStreamId(StreamToken id) {
		input_id = id;
		beginWork();
	}
	// void setOutputStreamId(StreamToken id) {
	// 	output_id = id;
	// 	beginWork();
	// }

	void beginWork() {
		// if (input_id == -1 || output_id == -1) return;
		if (input_id == -1) return;
		CkPrintf("Actually beginning work for Validators\n");
		CkPrintf("Validators getting from %d\n", input_id);


		Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Validators::recvData(0), thisProxy[thisIndex]));
		
	}

	void recvData(Ck::Stream::StreamDeliveryMsg* msg) {
		CkPrintf("---recvData on Validators\n");
		char* data = (char*)(msg -> data);
		if (msg->num_bytes != 0 && is_valid(data)) {
			CkPrintf("Data was valid! Got %d bytes\n", msg->num_bytes);
			CkPrintf("Got back %s\n", data);
			// Ck::Stream::putRecord(output_id, data, msg->num_bytes);
			Ck::Stream::flushLocalStream(output_id);
		} else {
			CkPrintf("Data wasn't valid!\n");
		}
		
		if (msg->status == Ck::Stream::StreamStatus::STREAM_OK) {
			Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Validators::recvData(0), thisProxy[thisIndex]));
		} else {
			CkCallback cb = CkCallback(CkReductionTarget(Validators, finishedTask), thisProxy[0]);
			contribute(cb);
		}
	}

	bool present_and_not_null(json j, std::string field) {
		return j.contains(field) && !j[field].is_null();
	}

	// return true if all fields present
	bool is_valid(char* cstr) {
		json j = json::parse(cstr);

		return (present_and_not_null(j, "name") && 
				present_and_not_null(j, "age") && 
				present_and_not_null(j, "biometrics") &&
				present_and_not_null(j["biometrics"], "heart_rate") && 
				present_and_not_null(j["biometrics"], "steps") && 
				present_and_not_null(j["biometrics"], "weight_kg") && 
				present_and_not_null(j["biometrics"], "height_m"));
	}
};

class Filters : public CBase_Filters {
	StreamToken input_id = -1;
	StreamToken output_id = -1; 
	CProxy_Main mainProxy;

public:
	Filters_SDAG_CODE
	Filters(CProxy_Main mainProxy): mainProxy(mainProxy) {}
	Filters(CkMigrateMessage* msg) {};
	void pup(PUP::er &p) {}

	void setInputStreamId(StreamToken id) {
		input_id = id;
		beginWork();
	}
	void setOutputStreamId(StreamToken id) {
		output_id = id;
		beginWork();
	}

	void beginWork() {
		if (input_id == -1 || output_id == -1) return;
		CkPrintf("Actually beginning work for Filters\n");
		Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Filters::recvData(0), thisProxy[thisIndex]));
	}
	void recvData(Ck::Stream::StreamDeliveryMsg* msg) {
		if (input_id == -1 || output_id == -1) return;
		char* data = (char*)(msg->data);
		if (!should_filter(data)) {
			Ck::Stream::putRecord(output_id, data, msg->num_bytes);
			Ck::Stream::flushLocalStream(output_id);
		}

		if (msg->status == Ck::Stream::StreamStatus::STREAM_OK) {
			Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Filters::recvData(0), thisProxy[thisIndex]));
		} else {
			CkCallback cb = CkCallback(CkReductionTarget(Filters, finishedTask), thisProxy[0]);
			contribute(cb);
		}
	}

	bool should_filter(char* cstr) {
		json j = json::parse(cstr);
		return (j["age"] < 25 || j["biometrics"]["heart_rate"] > 90);
	}
};

class Transformers : public CBase_Transformers {
	StreamToken input_id = -1;
	StreamToken output_id = -1;
	CProxy_Main mainProxy;

public:
	Transformers_SDAG_CODE
	Transformers(CProxy_Main mainProxy): mainProxy(mainProxy) {}
	Transformers(CkMigrateMessage* msg) {};
	void pup(PUP::er &p) {}

	void setInputStreamId(StreamToken id) {
		input_id = id;
		beginWork();
	}
	void setOutputStreamId(StreamToken id) {
		output_id = id;
		beginWork();
	}

	void beginWork() {
		if (input_id == -1 || output_id == -1) return;
		CkPrintf("Actually beginning work for Transformers\n");
		Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Transformers::recvData(0), thisProxy[thisIndex]));
	}
	void recvData(Ck::Stream::StreamDeliveryMsg* msg) {
		if (input_id == -1 || output_id == -1) return;
		char* data = (char*)(msg->data);
		std::string transformed_data = (transformJson(data));
		Ck::Stream::putRecord(output_id, (void*)transformed_data.c_str(), sizeof(char) * strlen(transformed_data.c_str()) + 1);
		Ck::Stream::flushLocalStream(output_id);

		if (msg->status == Ck::Stream::StreamStatus::STREAM_OK) {
			Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Transformers::recvData(0), thisProxy[thisIndex]));
		} else {
			CkCallback cb = CkCallback(CkReductionTarget(Transformers, finishedTask), thisProxy[0]);
			contribute(cb);
		}
	}

	double calculatePi(int num_iters) {
		double pi = 0.0;
		int i = 0;
		int sign = 1;
		while (i < num_iters) {
			pi += sign * (4.0 / (2 * i + 1));
			sign = -sign;
			++i;
		}
		return pi;
	}
	std::string transformJson(char* cstr) {
		json j = json::parse(cstr);
		j["height_in"] = j["height_m"].get<double>() * 39.3701;
		j["weight_lb"] = j["weight_kg"].get<double>() * 2.20462;
		j["pi"] = calculatePi(random_int(100000000, 500000000));
		return j.dump();
	}
};

class Writers : public CBase_Writers {
	StreamToken input_id = -1;
	std::string outfile_name = "";
	std::ofstream fd;
public:
	Writers_SDAG_CODE
	Writers() {
		std::ostringstream oss;
		oss << "outfile_" << thisIndex << ".txt";
		std::string outfile_name = oss.str();
		std::ofstream fd(outfile_name);
	}
	Writers(CkMigrateMessage* msg) {}
	void pup(PUP::er &p) {}

	void setInputStreamId(StreamToken id) {
		input_id = id;
		beginWork();
	}
	void beginWork() {
		if (input_id == -1) return;
		CkPrintf("Actually beginning work for Writers\n");
		Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Writers::recvData(0), thisProxy[thisIndex]));
	}

	void recvData(Ck::Stream::StreamDeliveryMsg* msg) {
		if (input_id == -1) return;
		fd << (char*)msg->data << "\n";


		if (msg->status == Ck::Stream::StreamStatus::STREAM_OK) {
			Ck::Stream::getRecord(input_id, CkCallback(CkIndex_Writers::recvData(0), thisProxy[thisIndex]));
		} else {
			fd.close();
			CkCallback cb = CkCallback(CkReductionTarget(Writers, finishedTask), thisProxy[0]);
			contribute(cb);
		}
	}
};

#include "streamtest.def.h"