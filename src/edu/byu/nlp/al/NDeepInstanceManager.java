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

import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.common.collect.Sets;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;

/**
 * An InstanceProvider that assigns each instance K annotations before moving to the next. The process is repeated
 * indefinitely if maxNumPasses==-1. Samples annotators WITHOUT replacement.
 * 
 * @author plf1
 *
 */
public class NDeepInstanceManager<D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

  public static NDeepInstanceManager<SparseFeatureVector, Integer> newManager(int k, int maxNumPasses, 
          Dataset data, AnnotationRecorder<SparseFeatureVector, Integer> annotationRecorder, RandomGenerator rnd) {
    List<FlatInstance<SparseFeatureVector, Integer>> instances = Datasets.instancesIn(data);  
      RandomRoundRobinQueue<FlatInstance<SparseFeatureVector, Integer>> q = RandomRoundRobinQueue.from(instances, k, maxNumPasses, rnd);
      return new NDeepInstanceManager<SparseFeatureVector, Integer>(q,annotationRecorder);
  }

	private final RandomRoundRobinQueue<FlatInstance<D, L>> q;
  private final PeekingIterator<FlatInstance<D, L>> it;
  private final Set<Integer> usedAnnotators= Sets.newConcurrentHashSet();
  private String previousInstanceSource = null;

	/** {@inheritDoc} */
	@Override
	protected boolean annotationAdded(FlatInstance<D, L> instance) {
	  usedAnnotators.add(instance.getAnnotator());
	  return true;
	}
	
	@VisibleForTesting
	public NDeepInstanceManager(RandomRoundRobinQueue<FlatInstance<D, L>> q, AnnotationRecorder<D, L> annotationRecorder) {
	  super(annotationRecorder);
		this.it = Iterators.peekingIterator(q.iterator());
		this.q=q;
	}
	
	@Override
  public FlatInstance<D, L> instanceFor(int annotatorId, long timeout, TimeUnit timeUnit)
	        throws InterruptedException {
	  synchronized (this) {
	    FlatInstance<D, L> inst = it.peek();
	    // clear usedAnnotators with each new (distinct) instance
	    if (!inst.getSource().equals(previousInstanceSource)){
	      usedAnnotators.clear();
	    }
	    // ignore annotators who have already been used for this instance
      if (usedAnnotators.contains(annotatorId)){
        return null;
      }
      previousInstanceSource = inst.getSource();
      usedAnnotators.add(annotatorId);
      return it.next();
    }
	}
	
	@Override
    public Iterable<FlatInstance<D, L>> getAllInstances() {
	    return q;
	}
	
	/** {@inheritDoc} */
	@Override
	public boolean isDone() {
		return !it.hasNext();
	}
}
