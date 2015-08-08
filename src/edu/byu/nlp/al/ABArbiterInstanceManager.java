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
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.Counters;
import edu.byu.nlp.util.Iterators2;

/**
 * An InstanceManager that annotates each item twice; in the case of disagreement, an arbiter provides a third and final
 * annotation.    
 * 
 * @author rah67
 *
 */
public abstract class ABArbiterInstanceManager<D, L> extends AbstractInstanceManager<D, L> implements InstanceManager<D, L> {

    protected abstract  List<FlatInstance<D,L>> getAnnotationsFor(String instanceSource);

    
    private final Collection<FlatInstance<D, L>> allInstances;
    // Iterator that repeats each instance twice for allInstances; synchronized in requestInstanceFor
    private final Iterator<FlatInstance<D, L>> it;
    private final BlockingDeque<FlatInstance<D, L>> conflicts;
    // Subset of allInstances for which there is consensus or an arbiter's decision
    private final Collection<Long> completed;
    private final Set<Long> arbiters;
    private final int numInstances;
    
    @VisibleForTesting
    ABArbiterInstanceManager(Collection<FlatInstance<D, L>> allInstances,
            Iterator<FlatInstance<D, L>> it, BlockingDeque<FlatInstance<D, L>> conflicts,
            Collection<Long> completed, Set<Long> arbiters, int numInstances,
            AnnotationRecorder<D,L> annotationRecorder) {
        super(annotationRecorder);
        this.allInstances = allInstances;
        this.it = it;
        this.conflicts = conflicts;
        this.completed = completed;
        this.arbiters = arbiters;
        this.numInstances = numInstances;
    }
  
    /** {@inheritDoc} **/
    @Override
    public boolean annotationAdded(FlatInstance<D, L> ann) {
      
      List<FlatInstance<D,L>> anns = getAnnotationsFor(ann.getSource());
      
        // For the first annotation, do not do anything special (the queue will serve it up as many times as
        // needed with new additional work here).
        if (anns.size() == 2) {
            // Do we need an arbiter?
            if (Counters.count(valuesOfAnnotations(anns)).numEntries() > 1) {
                conflicts.add(ann);
            } else {
                completed.add((long)ann.getInstanceId());
            }
        } else if (anns.size() == 3) {
            if (!arbiters.contains(anns.get(2).getAnnotator())) {
                throw new IllegalStateException("Expecting arbiter for 3rd annotation");
            }
            completed.add((long)ann.getInstanceId());
        } else if (anns.size() > 3) {
            throw new IllegalStateException("More than 3 annotations: " + ann);
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isDone() {
        return completed.size() == numInstances;
    }

    /** {@inheritDoc} */
    @Override
    public Collection<FlatInstance<D, L>> getAllInstances() {
        return allInstances;
    }

    /** {@inheritDoc} */
    @Override
    public FlatInstance<D, L> instanceFor(int annotatorId, long timeout, TimeUnit timeUnit) throws InterruptedException {
        if (arbiters.contains(annotatorId)) {
            return conflicts.poll(timeout, timeUnit);
        } else {
            // TODO(rhaertel): consider using a separate lock.
            synchronized (this) {
                if (it.hasNext()) {
                    return it.next();
                }
            }
        }
        return null;
    }
    
    

    /**
     * Shallow copy of instances.
     * 
     * @param twoPass when true, all instances will be annotated once before receiving second annotation.  
     */
    public static InstanceManager<SparseFeatureVector, Integer> newManager(final Dataset dataset, 
            boolean twoPass, Set<Long> arbiters) {
      
        List<FlatInstance<SparseFeatureVector, Integer>> allInstances = Datasets.instancesIn(dataset);
        Iterator<FlatInstance<SparseFeatureVector, Integer>> it;
        if (twoPass) {
            it = Iterators2.cycle(allInstances, 2);   
        } else {
            it = Iterators2.repeatItems(allInstances.iterator(), 2);
        }
        BlockingDeque<FlatInstance<SparseFeatureVector, Integer>> conflicts = new LinkedBlockingDeque<FlatInstance<SparseFeatureVector, Integer>>();
        Collection<Long> completed = Lists.newArrayListWithCapacity(allInstances.size());
        return new ABArbiterInstanceManager<SparseFeatureVector, Integer>(
            allInstances, it, conflicts, completed, arbiters, dataset.getInfo().getNumDocuments(), 
            new DatasetAnnotationRecorder(dataset)) {
          @Override
          protected List<FlatInstance<SparseFeatureVector, Integer>> getAnnotationsFor(String instanceSource) {
            return Lists.newArrayList(dataset.lookupInstance(instanceSource).getAnnotations().getRawAnnotations());
          }
        };
    }
    
    private Collection<L> valuesOfAnnotations(Collection<FlatInstance<D,L>> annotations){
      Collection<L> vals = Lists.newArrayList();
      for (FlatInstance<?,L> ann: annotations){
        vals.add(ann.getAnnotation());
      }
      return vals;
    }
    
    
}
