// chep-polyglot-jets.cc
//
// Code to run and time the jet finding of the "standard" 100 HepMC3
// events that we have used for the CHEP2023 Polyglot Jet Finding
// paper

#include "fastjet/ClusterSequence.hh"
#include <iostream> // needed for io
#include <cstdio>   // needed for io
#include <string>
#include <vector>
#include <chrono>

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/ReaderAscii.h"

using namespace std;
using Time = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

vector<vector<fastjet::PseudoJet>> read_input_events(const char* fname, long maxevents = -1) {
  // Read input events from a HepMC3 file, return the events in a vector
  // Each event is a vector of initial particles
  
  HepMC3::ReaderAscii input_file (fname);

  int events_parsed = 0;

  std::vector<std::vector<fastjet::PseudoJet>> events;

  while(!input_file.failed()) {
    if (maxevents >= 0 && events_parsed >= maxevents) break;

    std::vector<fastjet::PseudoJet> input_particles;

    HepMC3::GenEvent evt(HepMC3::Units::GEV, HepMC3::Units::MM);

    // Read event from input file
    input_file.read_event(evt);

    // If reading failed - exit loop
    if (input_file.failed()) break;

    ++events_parsed;
    input_particles.clear();
    input_particles.reserve(evt.particles().size());
    for(auto p: evt.particles()){
      if(p->status() == 1){
	      input_particles.emplace_back(p->momentum().px(),
				     p->momentum().py(),
				     p->momentum().pz(),
				     p->momentum().e());
      }
    }
    events.push_back(input_particles);
  }

  cout << "Read " << events_parsed << " events." << endl;
  return events;
}

vector<fastjet::PseudoJet> run_fastjet_clustering(std::vector<fastjet::PseudoJet> input_particles,
  fastjet::Strategy strategy) {
  // create a jet definition: a jet algorithm with a given radius parameter
  fastjet::RecombinationScheme recomb_scheme=fastjet::E_scheme;
  double R = 0.4;
  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R, recomb_scheme, strategy);


  // run the jet clustering with the above jet definition
  fastjet::ClusterSequence clust_seq(input_particles, jet_def);

  // // get the resulting jets ordered in pt
  double ptmin = 5.0;
  vector<fastjet::PseudoJet> inclusive_jets = sorted_by_pt(clust_seq.inclusive_jets(ptmin));

  return inclusive_jets;
}

int main(int argc, char* argv[]) {

  if (argc < 2 || argc > 6) {
    std::cout << "Usage: " << argv[0] << " <HepMC3_input_file> [max_events] [n_trials] [strategy] [dump]" << std::endl;
    std::cout << " Max events default is -1, which is all the events in the file" << std::endl;
    std::cout << " n_trials default is -1, which is the number of repeats to do" << std::endl;
    std::cout << " Max events default is -1, which is all the events in the file" << std::endl;
    std::cout << " strategy to use, valid values are 'Best' (default), 'N2Plain', 'N2Tiled'" << std::endl;
    std::cout << " dump - if non-zero, output jets are printed" << std::endl;
    exit(-1);
  }

  long maxevents = -1;
  long trials = 1;
  string mystrategy = "Best";
  bool dump = false;
  if (argc > 2) maxevents = strtoul(argv[2], 0, 0);
  if (argc > 3) trials = strtoul(argv[3], 0, 0);
  if (argc > 4) mystrategy = argv[4];
  if (argc > 5) dump = true;

  // read in input events
  //----------------------------------------------------------
  auto events = read_input_events(argv[1], maxevents);
  
  // Set strategy
  std::cout << mystrategy << endl;
  fastjet::Strategy strategy = fastjet::Best;
  if (mystrategy == string("N2Plain")) {
    strategy = fastjet::N2Plain;
  } else if (mystrategy == string("N2Tiled")) {
    strategy = fastjet::N2Tiled;
  }
  std::cout << mystrategy << endl;

  double time_total = 0.0;
  double time_total2 = 0.0;
  double sigma = 0.0;
  for (long trial = 0; trial < trials; ++trial) {
    std::cout << "Trial " << trial << " ";
    auto start_t = std::chrono::steady_clock::now();
    for (size_t ievt = 0; ievt < events.size(); ++ievt) {
      auto inclusive_jets = run_fastjet_clustering(events[ievt], strategy);

      if (dump) {
        std::cout << "Jets in processed event " << ievt+1 << ":" << endl;

      // // label the columns
      // printf("%5s %15s %15s %15s\n","jet #", "rapidity", "phi", "pt");
    
        // print out the details for each jet
        for (unsigned int i = 0; i < inclusive_jets.size(); i++) {
          printf("%5u %15.10f %15.10f %15.10f\n",
          i, inclusive_jets[i].rap(), inclusive_jets[i].phi(),
          inclusive_jets[i].perp());
        }
      }
    }
    auto stop_t = std::chrono::steady_clock::now();
    auto elapsed = stop_t - start_t;
    auto us_elapsed = double(chrono::duration_cast<chrono::microseconds>(elapsed).count());
    std::cout << us_elapsed << " us" << endl;
    time_total += us_elapsed;
    time_total2 += us_elapsed*us_elapsed;
  }
  time_total /= trials;
  time_total2 /= trials;
  if (trials > 1) {
    sigma = std::sqrt(double(trials)/(trials-1) * (time_total2 - time_total*time_total));
  } else {
    sigma = 0.0;
  }
  double mean_per_event = time_total / events.size();
  double sigma_per_event = sigma / events.size();
  std::cout << "Processed " << events.size() << " events, " << trials << " times" << endl;
  std::cout << "Total time " << time_total << " us" << endl;
  std::cout << "Time per event " << mean_per_event << " +- " << sigma_per_event << " us" << endl;

  return 0;
}
