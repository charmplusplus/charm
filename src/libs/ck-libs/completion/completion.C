#include "completion.h"

CompletionDetector::CompletionDetector()
{
    init();
}

void CompletionDetector::init() {
    producers_total = 0;
    producers_done_local = producers_done_global = 0;
    produced = 0;
    consumed = 0;
    running = false;
    unconsumed = 1; // Nonsense value, for loop below
}

void CompletionDetector::produce(int events_produced) {
    produced += events_produced;
}

void CompletionDetector::consume(int events_consumed) {
    consumed += events_consumed;
}

void CompletionDetector::done(int producers_done) {
    producers_done_local += producers_done;
}

#include "completion.def.h"
