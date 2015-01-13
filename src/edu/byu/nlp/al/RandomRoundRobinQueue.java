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
import com.google.common.collect.Lists;

import edu.byu.nlp.util.Collections3;
import edu.byu.nlp.util.Iterators2;

/**
 * Randomly returns each item in the queue exactly once then loops and repeats in a different random order each
 * time. This class is thread-safe. 
 */
// TODO(rhaertel): extract interface (or implement queue) and test!
public class RandomRoundRobinQueue<E> implements Iterable<E> {
    private final List<E> coll;
    private final RandomGenerator rnd;
    private Iterator<E> it;
    
    public static <E> RandomRoundRobinQueue<E> from(Iterable<E> source, int k, RandomGenerator rnd) {
        return new RandomRoundRobinQueue<E>(Lists.newArrayList(source), k, rnd);
    }
    
    @VisibleForTesting RandomRoundRobinQueue(List<E> coll, int k, RandomGenerator rnd) {
        this.coll = coll;
        this.rnd = rnd;
        this.it = Iterators2.repeatItems(coll.iterator(), k);
    }

    public synchronized E poll() {
        if (coll.size() == 0) {
            return null;
        }
        if (!it.hasNext()) {
            Collections3.shuffle(coll, rnd);
            it = coll.iterator();
        }
        return it.next();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized Iterator<E> iterator() {
        return Lists.newArrayList(coll).iterator();
    }
}