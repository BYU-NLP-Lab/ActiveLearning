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

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.annotations.VisibleForTesting;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;

/**
 * An InstanceProvider that assigns each instance K annotations before moving to the next. The process is repeated
 * indefinitely if maxNumPasses==-1. Samples annotators WITH replacement.
 * 
 * @author rah67
 *
 */
public class GeneralizedRoundRobinInstanceManager<D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

    public static GeneralizedRoundRobinInstanceManager<SparseFeatureVector, Integer> newManager(int k, 
            Dataset data, AnnotationRecorder<SparseFeatureVector, Integer> annotationRecorder, RandomGenerator rnd) {
      return newManager(k, -1, data, annotationRecorder, rnd);
    }
    
    public static GeneralizedRoundRobinInstanceManager<SparseFeatureVector, Integer> newManager(int k, int maxNumPasses, 
            Dataset data, AnnotationRecorder<SparseFeatureVector, Integer> annotationRecorder, RandomGenerator rnd) {
      List<FlatInstance<SparseFeatureVector, Integer>> instances = Datasets.instancesIn(data);  
        RandomRoundRobinQueue<FlatInstance<SparseFeatureVector, Integer>> q = RandomRoundRobinQueue.from(instances, k, maxNumPasses, rnd);
        return new GeneralizedRoundRobinInstanceManager<SparseFeatureVector, Integer>(q,annotationRecorder);
    }

  private Iterator<FlatInstance<D, L>> it;
	private final RandomRoundRobinQueue<FlatInstance<D, L>> q;

	@VisibleForTesting
	GeneralizedRoundRobinInstanceManager(RandomRoundRobinQueue<FlatInstance<D, L>> q, AnnotationRecorder<D, L> annotationRecorder) {
	  super(annotationRecorder);
		this.it = q.iterator();
		this.q=q;
	}
	
	@Override
    public FlatInstance<D, L> instanceFor(long annotatorId, long timeout, TimeUnit timeUnit)
	        throws InterruptedException {
	    return it.next();
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
