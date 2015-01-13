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
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.annotationinterface.java.AnnotationInterfaceJavaUtils;
import edu.byu.nlp.data.FlatAnnotatedInstance;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.util.FutureIterator;
import edu.byu.nlp.util.ResettableBooleanLatch;

/**
 * @author rah67
 */
public abstract class AbstractInstanceManager<D, L> implements InstanceManager<D, L> {
  
    /**
     * Hook to allow for processing after the annotation has already added to the instance.
     */
    protected boolean annotationAdded(FlatInstance<D, L> instance) {
        return true;
    }
    
    protected void cancelRequest(FlatInstance<D, L> instance, long annotatorId) {
        throw new UnsupportedOperationException();
    }
    
    /**
     * This method should usually not be synchronized as it is called by the synchronized method instances().
     */
    protected abstract Iterable<FlatInstance<D, L>> getAllInstances();

    /**
     * Returns null if there are currently no instances available for the annotator, or a timeout occurs. 
     */
    protected abstract FlatInstance<D, L> instanceFor(long annotatorId, long timeout, TimeUnit timeUnit) 
            throws InterruptedException;

    
    
    
//    List<FlatAnnotatedInstance<D, L>> annotations = Lists.newArrayList();
    private int numAnnotations = 0;
    private AnnotationRecorder<D, L> annotationRecorder;
    public AbstractInstanceManager(AnnotationRecorder<D,L> annotationRecorder){
      this.annotationRecorder=annotationRecorder;
    }

    /** {@inheritDoc} */
    @Override
    public final synchronized AnnotationRequest<D, L> requestInstanceFor(long annotatorId, long timeout, TimeUnit timeUnit) throws InterruptedException {
        FlatInstance<D, L> instance = instanceFor(annotatorId, timeout, timeUnit);
        return instance == null ? null : new BasicAnnotationRequest(annotatorId, instance);
    }

    /** {@inheritDoc} */
    @Override
    public final synchronized Collection<FlatInstance<D, L>> instances() {
      return Lists.newArrayList(getAllInstances());
    }

    @Override
    public long numAnnotations() {
//        return annotations.size();
        return numAnnotations;
    }
    
    private final List<InstanceFutureIterator> iterators = Collections.synchronizedList(
            Lists.<InstanceFutureIterator>newArrayList());

    private class InstanceFutureIterator implements FutureIterator<Collection<FlatInstance<D, L>>> {

        private final ResettableBooleanLatch latch = new ResettableBooleanLatch();
        private volatile boolean done = false;
        
        /** {@inheritDoc} */
        @Override
        public Collection<FlatInstance<D, L>> next() throws InterruptedException {
            if (done) {
                return null;
            }
            // FIXME(rhaertel): b/t isDone() and await(), the last signal can happen and we would be
            // stuck forever in this await.
            if (!isDone()) {
                latch.await();
            } else {
                done = true;
            }
            Collection<FlatInstance<D, L>> data = instances();
            // It is possible for data to contain the whole data set at this point, in which case
            // it will be returned twice which shouldn't be a problem.
            if (!isDone()) {
                // It is possible for isDone to flip to true before a reset is called, but then the await won't be
                // called on the next time through.
                latch.reset();
            }
            return data;
        }

        /** {@inheritDoc} 
         * @throws InterruptedException */
        @Override
        public Collection<FlatInstance<D, L>> next(long timeout, TimeUnit unit) throws InterruptedException {
            if (done) {
                return null;
            }
            if (latch.await(timeout, unit)) {
                Collection<FlatInstance<D, L>> data = instances();
                latch.reset();
                return data;
            } else if (isDone()) {
                done = true;
                // Don't reset
                // We will return the instances one last time to ensure the client gets all of the data since it is
                // possible to reach this point with some updates to instances since last call to next().
                return instances();
            }
            // There was no data ready in time, but there still is data to look forward to.
            return null;
        }
        
        public ResettableBooleanLatch getLatch() {
            return latch;
        }
    }

    /**
     * Creates a new future iterator.
     */
    public FutureIterator<Collection<FlatInstance<D, L>>> newInstanceFutureIterator() {
        if (isDone()) {
            return null;
        }
        InstanceFutureIterator it = new InstanceFutureIterator();
        iterators.add(it);
        return it;
    }

    /**
     * Opens all latches, i.e., unblocks threads blocked on .next().
     */
    private void signalAllIterators() {
        synchronized (iterators) {
            boolean done = isDone();
            Iterator<InstanceFutureIterator> it = iterators.iterator();
            while (it.hasNext()) {
                it.next().getLatch().signal();
                if (done) {
                    // We need to unregister iterators once iteration has terminated to prevent memory leaks.
                    it.remove();
                }
            }
        }
    }
    
    

    private class BasicAnnotationRequest implements AnnotationRequest<D, L> {

        private final long annotatorId;
        private final FlatInstance<D, L> instance;
        
        public BasicAnnotationRequest(long annotatorId, FlatInstance<D, L> instance) {
            this.annotatorId = annotatorId;
            this.instance = instance;
        }

        /** {@inheritDoc} */
        @Override
        public long getAnnotatorId() {
            return annotatorId;
        }

        /** {@inheritDoc} */
        @Override
        public FlatInstance<D, L> getInstance() {
            return instance;
        }

        /** {@inheritDoc} */
        @Override
        public void cancelRequest() {
            throw new UnsupportedOperationException("Cannot cancel request");
        }

        /** {@inheritDoc} */
        @Override
        public boolean storeAnnotation(AnnotationInfo<L> annotationInfo) {
          
          FlatAnnotatedInstance<D, L> annotation = new FlatAnnotatedInstance<D,L>(
              AnnotationInterfaceJavaUtils.<D,L>newAnnotatedInstance(
                  annotatorId, 
                  annotationInfo.getAnnotation(), 
                  annotationInfo.getAnnotationEvent().getStartTimeNanos(), 
                  annotationInfo.getAnnotationEvent().getEndTimeNanos(), 
                  instance.getInstanceId(),
                  instance.getSource(),  
                  instance.getData() 
                  ));
          Preconditions.checkState(annotation.getInstanceId() == instance.getInstanceId(),
              "Reconstituted instance id "+annotation.getInstanceId()+" does not match the original "+instance.getInstanceId());
          
//          annotations.add(annotation);
          numAnnotations++;
          
          // delegate the job of actually recording the annotation to a concrete implementation
//          try{
            annotationRecorder.recordAnnotation(annotation);
            annotationAdded(annotation);          
            signalAllIterators();
            return true;
//          }
//          catch(Exception e){
//            logger
//            return false;
//          }
        }
    }
    
    
    public static interface AnnotationRecorder<D,L> {
      void recordAnnotation(FlatInstance<D,L> annotation);
    }
    
}