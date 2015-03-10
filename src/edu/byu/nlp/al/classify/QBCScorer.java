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
package edu.byu.nlp.al.classify;

import java.util.List;
import java.util.Map.Entry;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import edu.byu.nlp.al.Scorer;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.stats.CategoricalDistribution;
import edu.byu.nlp.stats.ConditionalCategoricalDistribution;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DenseCounter;

/**
 * Notes: age is the age of the *oldest* model.
 * @author rah67
 *
 */
public class QBCScorer implements Scorer {

    public static interface DisagreementCalculator {
        double disagreementOf(List<CategoricalDistribution> labelDists);
    }
    
    public static class VoteEntropy implements DisagreementCalculator {

        private final int numLabels;
        
        public VoteEntropy(int numLabels) {
            this.numLabels = numLabels;
        }
        
        /** {@inheritDoc} */
        @Override
        public double disagreementOf(List<CategoricalDistribution> labelDists) {
            Counter<Integer> counts = new DenseCounter(numLabels);
            for (CategoricalDistribution dist : labelDists) {
                counts.incrementCount(dist.argMax(), 1);
            }
            assert labelDists.size() == counts.totalCount();
            double logTotal = Math.log(labelDists.size());
            double entropy = 0.0;
            for (Entry<Integer, Integer> entry : counts.entrySet()) {
                if (entry.getValue() > 0) {
                    entropy -= entry.getValue() * (Math.log(entry.getValue()) - logTotal);
                }
            }
            return entropy / labelDists.size();
        }
        
    }
    
    private static final class AgedDistribution {
        private final ConditionalCategoricalDistribution<SparseFeatureVector> dist;
        private final long age;

        public AgedDistribution(ConditionalCategoricalDistribution<SparseFeatureVector> dist, long age) {
            this.dist = dist;
            this.age = age;
        }

        public ConditionalCategoricalDistribution<SparseFeatureVector> getDist() {
            return dist;
        }

        public long getAge() {
            return age;
        }
    }
    
    private final DisagreementCalculator calc;
    private List<AgedDistribution> dists;
    
    private static class DistToAgedDist implements
            Function<ConditionalCategoricalDistribution<SparseFeatureVector>, AgedDistribution> {
        
        private final long age;
        
        public DistToAgedDist(long age) {
            this.age = age;
        }
        
        @Override
        public AgedDistribution apply(ConditionalCategoricalDistribution<SparseFeatureVector> dist) {
            return new AgedDistribution(dist, age);
        }
    }
    
    public static QBCScorer from(DisagreementCalculator calc,
                                 Iterable<? extends ConditionalCategoricalDistribution<SparseFeatureVector>> dists) {
        List<AgedDistribution> agedDists = Lists.newArrayList(Iterables.transform(dists, new DistToAgedDist(0)));
        return new QBCScorer(calc, agedDists);
    }
    
    /**
     * Copies the list.
     */
    @VisibleForTesting QBCScorer(DisagreementCalculator calc, List<AgedDistribution> dists) {
        this.calc = calc;
        this.dists = dists;
    }

    public void setDist(int index, ConditionalCategoricalDistribution<SparseFeatureVector> dist, long age) {
        dists.set(index, new AgedDistribution(dist, age));
    }

    /** {@inheritDoc} */
    @Override
    public Score score(FlatInstance instance) {
//      List<CategoricalDistribution> labelDists = Lists.newArrayListWithCapacity(dists.size());
//      long age = Long.MAX_VALUE;
//      for (AgedDistribution dist : dists) {
//          labelDists.add(dist.getDist().given(instance.asFeatureVector()));
//          if (dist.getAge() < age) {
//              age = dist.getAge();
//          }
//      }
//      return new Score(calc.disagreementOf(labelDists), age);
      // FIXME (pfelt): see note in LabelUncertaintyScorer
      return null;
    }
}
