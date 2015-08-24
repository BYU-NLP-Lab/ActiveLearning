package edu.byu.nlp.al;

import java.util.Collection;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;

public interface MeasurementSelector {
  public Collection<FlatInstance<SparseFeatureVector, Integer>> selectNext(int batchSize);
}
