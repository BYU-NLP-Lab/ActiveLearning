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

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;

import edu.byu.nlp.util.Collections3;
import edu.byu.nlp.util.Iterators2;

/**
 * Randomly returns each item in the queue exactly once then loops and repeats in a different random order each
 * time. This class is thread-safe. 
 */
public class RandomRoundRobinQueue<E> implements Iterable<E> {
 
    private final List<E> coll;
    private final RandomGenerator rnd;
    private int maxNumRepetitions;
    private int k;

    /**
     * Loops forever
     */
    public static <E> RandomRoundRobinQueue<E> from(Iterable<E> source, int k, RandomGenerator rnd) {
      return from(source,k,-1,rnd);
    }
    
    /**
     * Loops maxNumRepetitions times
     */
    public static <E> RandomRoundRobinQueue<E> from(Iterable<E> source, int k, int maxNumRepetitions, RandomGenerator rnd) {
        return new RandomRoundRobinQueue<E>(Lists.newArrayList(source), k, maxNumRepetitions, rnd);
    }
    
    @VisibleForTesting RandomRoundRobinQueue(List<E> coll, int k, int maxNumRepetitions, RandomGenerator rnd) {
        this.coll = coll;
        this.rnd = rnd;
        this.maxNumRepetitions=maxNumRepetitions;
        this.k=k;
    }


    /** {@inheritDoc} */
    @Override
    public synchronized Iterator<E> iterator() {
      return new AbstractIterator<E>(){
        @Override
        protected E computeNext() {
          try{
            return poll();
          }
          catch (NoMoreItemsException e){
            return endOfData();
          }
        }

        private Iterator<E> it = null;
        private int numRepetitions = 0;
        private synchronized E poll() throws NoMoreItemsException {
            if (coll.size() == 0) {
              throw new NoMoreItemsException(); // signal finished
            }
            if (it==null || !it.hasNext()) {
              if (maxNumRepetitions>0 && numRepetitions>=maxNumRepetitions){
                throw new NoMoreItemsException(); // signal finished
              }
              else{
                Collections3.shuffle(coll, rnd);
                it = Iterators2.repeatItems(coll.iterator(), k);
                numRepetitions++;
              }
            }
            return it.next();
        }
      };
    }
    
    @SuppressWarnings("serial")
    private static class NoMoreItemsException extends Exception{}
}