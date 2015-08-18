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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.measurements.ClassificationMeasurements.BasicClassificationLabelProportionMeasurement;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.Deques;
import edu.byu.nlp.util.Iterables2;

/**
 * @author pfelt
 * 
 * Uses empirical annotations but does not respect their order. Annotates the entire dataset once, then twice, and so on 
 * until there are no more annotations. Even if some instances are annotated more than others, they are build up in layers.
 * Which annotations are chosen for the first layer, etc., is subject to the random seed and not to timestamps. 
 *
 */
public class EmpiricalAnnotationLayersInstanceManager <D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

  public List<FlatInstance<D, L>> queue;
  
  
  @VisibleForTesting
  EmpiricalAnnotationLayersInstanceManager(Iterable<FlatInstance<D, L>> instances, EmpiricalAnnotations<D, L> annotations,
      AnnotationRecorder<D, L> annotationRecorder, int maxNumAnnotations, int maxNumMeasurements, boolean prioritizeLabelProportions, RandomGenerator rnd) {
    super(annotationRecorder);
    
    // make a mutable collection of all annotations for each instance
    List<FlatInstance<D, L>> sortedAnnotations = Lists.newArrayList();
    Map<String,Deque<FlatInstance<D, L>>> perInstanceAnnotationLists = Maps.newIdentityHashMap(); 
    for (FlatInstance<D, L> inst: instances){
      // find all annotations associated with this item
      Collection<FlatInstance<D, L>> anns = annotations.getAnnotationsFor(inst.getSource(), inst.getData()).values();
      perInstanceAnnotationLists.put(inst.getSource(), Deques.randomizedDeque(anns, rnd));
    }
    
    // grab one annotation for each instance until they are gone
    // (annotate the whole corpus 1-deep before starting on 2-deep, and so on)
    while(perInstanceAnnotationLists.size()>0){
      Set<String> toRemove = Sets.newHashSet();
      
      for (String src: Iterables2.shuffled(perInstanceAnnotationLists.keySet(), rnd)){
          Deque<FlatInstance<D, L>> anns = perInstanceAnnotationLists.get(src);
          if (anns.size()>0){
            // add 1 to the queue for this instance
            sortedAnnotations.add(anns.pop());
          }
          if (anns.size()==0){
            toRemove.add(src);
          }
      }
      
      for (String src: toRemove){
        perInstanceAnnotationLists.remove(src);
      }
    } 
    

    // interleave measurements and annotations in the final queue
    Deque<FlatInstance<D,L>> measurementDeque = Deques.randomizedDeque(annotations.getMeasurements(), rnd);
    prioritizeMeasurements(measurementDeque, prioritizeLabelProportions);
    Deque<FlatInstance<D,L>> annotationDeque = new ArrayDeque<FlatInstance<D,L>>(sortedAnnotations);
    queue = Lists.newLinkedList(); // better queueing behavior

    // add measurements 
    int numMeasurements = 0;
    while(measurementDeque.size()>0 && numMeasurements<maxNumMeasurements){
      numMeasurements += 1;
      queue.add(measurementDeque.pop());
    }

    // add annotations 
    int numAnnotations = 0;
    while(annotationDeque.size()>0 && numAnnotations<maxNumAnnotations){
      numAnnotations += 1;
      queue.add(annotationDeque.pop());
    }
    
  }
  
  
  @Override
    public FlatInstance<D, L> instanceFor(int annotatorId, long timeout, TimeUnit timeUnit)
          throws InterruptedException {
    
    if (queue.size()>0){
      // get next most recent annotation in the queue
      FlatInstance<D, L> nextann = queue.get(0);
      
      // if it belongs to annotatorId, return it
      if (nextann.getAnnotator()==annotatorId){
        queue.remove(nextann);
        return nextann;
      }
    }
    
    // return null if annotatorId is not the next one who historically 
    // gave an annotation.
    // This strategy of try+fail will incur a time cost,
    // but the learning curve driver class grabs random 
    // annotators as quickly as it can and there aren't usually too many, 
    // so the penalty won't be too severe
    return null;
  }
  
  @Override
  public Collection<FlatInstance<D, L>> getAllInstances() {
    return queue;
  }
  
  /** {@inheritDoc} */
  @Override
  public boolean isDone() {
    return queue.size()==0;
  }
  

  public static EmpiricalAnnotationLayersInstanceManager<SparseFeatureVector, Integer> newManager(
          Dataset dataset, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations, int maxNumAnnotations,
          int maxNumMeasurements, boolean prioritizeLabelProportions, RandomGenerator rnd) {
    
    List<FlatInstance<SparseFeatureVector, Integer>> instances = Datasets.instancesIn(dataset);
    return new EmpiricalAnnotationLayersInstanceManager<SparseFeatureVector, Integer>(instances,annotations,
          new DatasetAnnotationRecorder(dataset), maxNumAnnotations, maxNumMeasurements, prioritizeLabelProportions, rnd);
  }
  
  

  
  private void prioritizeMeasurements(Deque<FlatInstance<D, L>> measurementDeque, boolean prioritizeLabelProportions) {
    if (prioritizeLabelProportions){
      // get a list of all the labeled proportion measurements 
      ArrayList<FlatInstance<D, L>> proportions = Lists.newArrayList();
      for (FlatInstance<D, L> inst: measurementDeque){
        Measurement meas = inst.getMeasurement();
        if (meas instanceof BasicClassificationLabelProportionMeasurement){
          proportions.add(inst);
        }
      }
      // move them to the front of the line
      measurementDeque.removeAll(proportions);
      for (FlatInstance<D, L> prop: proportions){
        measurementDeque.push(prop);
      }
    }
  }

}