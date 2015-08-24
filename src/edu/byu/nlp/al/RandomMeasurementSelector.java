/**
 * Copyright 2015 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.al;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.random.RandomAdaptor;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementModelBuilder;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;

/**
 * @author plf1
 *
 */
public class RandomMeasurementSelector implements MeasurementSelector {
  private static Logger logger = LoggerFactory.getLogger(RandomMeasurementSelector.class);
  
  private List<FlatInstance<SparseFeatureVector, Integer>> candidates = Lists.newArrayList();

  /**
   * @param modelBuilder
   * @param annotations 
   */
  public RandomMeasurementSelector(
      MeasurementModelBuilder modelBuilder, Dataset dataset, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations, RandomGenerator rnd) {
    // we want to add all measurements that are not already taken (used as seed set contained in dataset)
    // FIXME: this is horrifically inefficient! Fix it! 
    for (FlatInstance<SparseFeatureVector, Integer> meas: annotations.getMeasurements()){
      if (!dataset.getMeasurements().contains(meas)){
        candidates.add(meas);
      }
    }
    for (Multimap<Integer, FlatInstance<SparseFeatureVector, Integer>> perAnnotatorAnnotations: annotations.getPerInstancePerAnnotatorAnnotations().values()){
      for (FlatInstance<SparseFeatureVector, Integer> meas: perAnnotatorAnnotations.values()){
        candidates.add(meas);
      }
    }
    Collections.shuffle(candidates, new RandomAdaptor(rnd));
  }
  
  public Collection<FlatInstance<SparseFeatureVector, Integer>> selectNext(int batchSize){

    List<FlatInstance<SparseFeatureVector, Integer>> batch = Lists.newArrayList(candidates.subList(0, Math.min(batchSize,candidates.size())));
    candidates.removeAll(batch);
    logger.info("\n**********************************************************\n"
        + "******* Selected batch of size "+batch.size()+" *******\n"
        + "**********************************************************\n");
    return batch;
  }
  
  
}
