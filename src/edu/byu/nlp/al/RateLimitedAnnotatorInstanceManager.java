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
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.base.Preconditions;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.util.FutureIterator;

/**
 * @author plf1
 * 
 * An instance manager that wraps another instance manager and   
 * returns requests for specific annotators according a given 
 * fraction of the time (other times returns null for that annotator)
 */
public class RateLimitedAnnotatorInstanceManager<D, L> implements InstanceManager<D, L> {

  private InstanceManager<D, L> delegate;
  private Map<Integer, Double> annotatorRates;
  private RandomGenerator rnd;

  public RateLimitedAnnotatorInstanceManager(InstanceManager<D, L> delegate, Map<Integer, Double> annotatorRates, RandomGenerator rnd){
    Preconditions.checkNotNull(delegate);
    Preconditions.checkNotNull(annotatorRates);
    Preconditions.checkNotNull(rnd);
    this.delegate=delegate;
    this.annotatorRates=annotatorRates;
    this.rnd=rnd;
  }

  /** {@inheritDoc} */
  @Override
  public AnnotationRequest<D, L> requestInstanceFor(int annotatorId, long timeout, TimeUnit timeUnit) throws InterruptedException {
    Preconditions.checkArgument(annotatorRates.containsKey(annotatorId));
    if (rnd.nextDouble()<annotatorRates.get(annotatorId)){
      return delegate.requestInstanceFor(annotatorId, timeout, timeUnit);
    }
    else{
      return null;
    }
  }

  /** {@inheritDoc} */
  @Override
  public Iterable<FlatInstance<D, L>> instances() {
    return delegate.instances();
  }

  /** {@inheritDoc} */
  @Override
  public boolean isDone() {
    return delegate.isDone();
  }

  /** {@inheritDoc} */
  @Override
  public FutureIterator<Collection<FlatInstance<D, L>>> newInstanceFutureIterator() {
    return delegate.newInstanceFutureIterator();
  }

  /** {@inheritDoc} */
  @Override
  public long numAnnotations() {
    return delegate.numAnnotations();
  }

}
