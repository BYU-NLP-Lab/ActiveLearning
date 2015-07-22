/**
 * Copyright 2012 Brigham Young University
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
import java.util.List;
import java.util.concurrent.TimeUnit;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;

/**
 * An InstanceProvider that reveals existing annotations based on their timestamps so 
 * that order they were collected is preserved. 
 * 
 * @author pfelt
 *
 */
public class EmpiricalAnnotationInstanceManager<D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

  private List<FlatInstance<D, L>> queue;


  @VisibleForTesting
	EmpiricalAnnotationInstanceManager(Iterable<FlatInstance<D, L>> instances, EmpiricalAnnotations<D, L> annotations,
	    AnnotationRecorder<D, L> annotationRecorder) {
    super(annotationRecorder);
	  queue = Lists.newArrayList();
	  for (FlatInstance<D, L> inst: instances){
	    // add each annotation associated with this item to the queue 
	    queue.addAll(annotations.getAnnotationsFor(inst.getInstanceId(), inst.getData()).values());
	  }
	  // sort the annotation queue based on annotation order
	  Datasets.sortAnnotations(queue);
	  
	  queue = Lists.newLinkedList(queue); // better queueing behavior
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
	

  public static EmpiricalAnnotationInstanceManager<SparseFeatureVector, Integer> newManager(
          Dataset dataset, EmpiricalAnnotations<SparseFeatureVector, Integer> annotations) {
    
    List<FlatInstance<SparseFeatureVector, Integer>> instances = Datasets.instancesIn(dataset);
    return new EmpiricalAnnotationInstanceManager<SparseFeatureVector, Integer>(instances,annotations,
          new DatasetAnnotationRecorder(dataset));
  }
  
}
