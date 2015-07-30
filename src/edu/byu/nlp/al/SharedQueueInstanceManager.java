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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import edu.byu.nlp.al.Scorer.Score;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.BlockingSortedListHeap;
import edu.byu.nlp.util.Heap;
import edu.byu.nlp.util.Heap.HeapIterator;

/**
 * An InstanceManager uses a Scorer to maintain a shared queue from which instances are selected. 
 * 
 * @author rah67
 * @author plf1
 *
 */
public class SharedQueueInstanceManager<D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

	  private static final Logger logger = LoggerFactory.getLogger(SharedQueueInstanceManager.class);
    
	private final BlockingSortedListHeap<ScoredFlatInstance<D, L>> q;
	private final Iterable<FlatInstance<D, L>> allInstances;
	
	@VisibleForTesting
	SharedQueueInstanceManager(BlockingSortedListHeap<ScoredFlatInstance<D, L>> q,
	                           Iterable<FlatInstance<D,L>> allInstances, AnnotationRecorder<D, L> annotationRecorder) {
	  super(annotationRecorder);
		this.q = q;
		this.allInstances = allInstances;
	}
	
	private static final class DefaultScore<D, L> implements Function<FlatInstance<D, L>, ScoredFlatInstance<D, L>> {
	    @Override
	    public ScoredFlatInstance<D, L> apply(FlatInstance<D, L> instance) {
	        return ScoredFlatInstance.from(instance, Double.POSITIVE_INFINITY, -1);
	    }
	}
	
	// This class assumes ownership of Instances; in particular, instance.getAnnotations() should not be modified.
	public static SharedQueueInstanceManager<SparseFeatureVector, Integer> newProvider(Dataset dataset,
	        Scorer<SparseFeatureVector, Integer> scorer, boolean recordMeasurements) {
	  
	  List<FlatInstance<SparseFeatureVector, Integer>> instances = Datasets.annotationsIn(dataset);
	  
    // Rather than using min-first sorting, we will avoid the extra overhead by just manually negating the score in
    // the scoring thread.
    BlockingSortedListHeap<ScoredFlatInstance<SparseFeatureVector, Integer>> q =
            BlockingSortedListHeap.from(Iterables.transform(instances, new DefaultScore<SparseFeatureVector, Integer>()));
      new Thread(new ScoringThread<SparseFeatureVector, Integer>(q, scorer), "Scorer").start();
    return new SharedQueueInstanceManager<SparseFeatureVector, Integer>(q, instances, 
        new DatasetAnnotationRecorder(dataset, recordMeasurements));
	}

	/** {@inheritDoc} */
	@Override
	public boolean isDone() {
	    return false;
	}

    /** {@inheritDoc} */
    @Override
    public Collection<FlatInstance<D, L>> getAllInstances() {
        return Lists.newArrayList(allInstances);
    }

    /** {@inheritDoc} */
    @Override
    public FlatInstance<D, L> instanceFor(int annotatorId, long timeout, TimeUnit timeUnit) throws InterruptedException {
        ScoredFlatInstance<D, L> ScoredFlatInstance = q.poll(timeout, timeUnit);
        if (ScoredFlatInstance != null) {
            long staleness = numAnnotations() - ScoredFlatInstance.getAge();
            logger.info(String.format("Staleness = %d (Age = %d)", staleness, ScoredFlatInstance.getAge()));
            return ScoredFlatInstance.getInstance();
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean annotationAdded(FlatInstance<D, L> instance) {
        return q.offer(ScoredFlatInstance.from(instance, Double.POSITIVE_INFINITY, -1));
    }
    
    

    public static class ScoredFlatInstance<D, L> implements Comparable<ScoredFlatInstance<D, L>> {
        private final FlatInstance<D, L> instance;
        private double score;
        private long age;
        
        public ScoredFlatInstance(FlatInstance<D, L> instance, double score, long age) {
            this.instance = instance;
            this.score = score;
            this.age = age;
        }

        public FlatInstance<D, L> getInstance() { return instance; }
        public double getScore() { return score; }
        public long getAge() { return age; }

        public static <D, L> ScoredFlatInstance<D, L> from(FlatInstance<D, L> instance, double score, long age) {
            return new ScoredFlatInstance<D, L>(instance, score, age);
        }

        @Override
        public int compareTo(ScoredFlatInstance<D, L> that) {
            return Double.compare(this.score, that.score);
        }
    }
    
    public static class ScoringThread<D, L> implements Runnable {

        private final Heap<ScoredFlatInstance<D, L>> q;
        private final Scorer<D, L> scorer;
        
        public ScoringThread(Heap<ScoredFlatInstance<D, L>> q, Scorer<D, L> scorer) {
            this.q = q;
            this.scorer = scorer;
        }
        
        /** {@inheritDoc} */
        @Override
        public void run() {
            while (true) {
                HeapIterator<ScoredFlatInstance<D, L>> it = q.iterator();
                while (it.hasNext()) {
                    ScoredFlatInstance<D, L> item = it.next();
                    Score newScore = scorer.score(item.getInstance());
                    // Rather than using min-first sorting, we will avoid the extra overhead by just manually negating
                    // the score.
                    it.replace(ScoredFlatInstance.from(item.getInstance(), -newScore.getScore(), newScore.getBirthDate()));
                }
            }
        }
    }
    
}
