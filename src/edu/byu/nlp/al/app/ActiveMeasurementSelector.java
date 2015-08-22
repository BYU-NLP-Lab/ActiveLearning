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
package edu.byu.nlp.al.app;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;

import edu.byu.nlp.crowdsourcing.measurements.MeasurementModelBuilder;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;

/**
 * @author plf1
 *
 */
public class ActiveMeasurementSelector {

//  private Set<FlatInstance<SparseFeatureVector, Integer>> used = Sets.newHashSet();
//  private Set<FlatInstance<SparseFeatureVector, Integer>> candidates = Sets.newHashSet();
  private Deque<FlatInstance<SparseFeatureVector, Integer>> candidates = new ArrayDeque(); // Lists.newArrayList();
  private MeasurementModelBuilder modelBuilder;
//  private EmpiricalAnnotations<SparseFeatureVector, Integer> annotations;

  /**
   * @param modelBuilder
   * @param annotations 
   */
  public ActiveMeasurementSelector(MeasurementModelBuilder modelBuilder, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations) {
    this.modelBuilder=modelBuilder;
//    this.annotations=annotations;
//    candidates.addAll(annotations.getMeasurements());
    for (Multimap<Integer, FlatInstance<SparseFeatureVector, Integer>> perAnnotatorAnnotations: annotations.getPerInstancePerAnnotatorAnnotations().values()){
      candidates.addAll(perAnnotatorAnnotations.values());
    }
  }
  
  public Collection<FlatInstance<SparseFeatureVector, Integer>> selectNext(int batchSize){
    List<FlatInstance<SparseFeatureVector, Integer>> retval = Lists.newArrayList(); 
    for (int i=0; i<batchSize; i++){
      if (candidates.size()>0){
        retval.add(candidates.pop());
      }
    }
//    for (FlatInstance<SparseFeatureVector, Integer> meas: candidates){
//      if (!used.contains(meas)){
//        used.add(meas);
//        return meas;
//      }
//    }
//    return null;
    return retval;
  }

}
