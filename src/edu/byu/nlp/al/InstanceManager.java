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
import java.util.concurrent.TimeUnit;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.util.FutureIterator;


/**
 * @author rah67
 *
 */
public interface InstanceManager<D, L> {

	/** Provides the next instance for the indicated annotator. **/
	AnnotationRequest<D, L> requestInstanceFor(int annotatorId, long timeout, TimeUnit timeUnit) throws InterruptedException;
	Iterable<FlatInstance<D, L>> instances();

	boolean isDone();
	
	/**
	 * Creates a new FutureIterator which can be used to wait for new instances to become available and then grab a
	 * consistent copy of the data at that time. Between calls to next, multiple updates can be made to the dataset
	 * in which case they are all "batched together" and available all together on the next invocation. Example usage:
	 *  <pre>
	 * 	  public static class Consumer<D, L> implements Runnable {
	 *	    @Override
	 *	    public void run() {
	 *		  FutureIterator<Collection<FlatInstance<D, L>>> future = im.futureIterator();
	 *		  while (true) {
	 *		    Collection<FlatInstance<D, L>> instances = future.next();
	 *		    learner.train(instances);
	 *	      }
	 *	    }
	 *   }
	 *   </pre>
	 */
	// TODO(rah67): we need to do a virtual typedef here to shorten the generics
	FutureIterator<Collection<FlatInstance<D, L>>> newInstanceFutureIterator();
	
	long numAnnotations();
}
