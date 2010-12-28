// One instance is created and called on each PE
void KMeansGroup::cluster()
{
  CLUSTERS::Write w = clusters.getInitialWrite();
  if (initSeed != -1) writePosition(w, initSeed);

  // Put the array in Read mode
  CLUSTERS::Read r = w.syncToRead();

  do {
    // Each PE finds the seed closest to itself
    double minDistance = distance(r, curSeed);
    
    for (int i = 0; i < numClusters; ++i) {
      double d = distance(r, i);
      if(d < minDistance) {
	minDistance = d;
	newSeed = i;
      }
    }

    // Put the array in Accumulate mode, 
    // excluding the current value
    CLUSTERS::Accum a = r.syncToExcAccum();
    // Each PE adds itself to its new seed
    for (int i = 0; i < numMetrics; ++i)
      a(newSeed, i) += metrics[i];

    // Update membership and change count
    a(newSeed, numMetrics) += 1;
    if (curSeed != newSeed)
      a(0, numMetrics+1) += 1;
    curSeed = newSeed;
      
    // Put the array in Read mode
    r = a.syncToRead();
  } while(r(0, numMetrics+1) > 0);
}
