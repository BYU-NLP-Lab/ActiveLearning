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

import edu.byu.nlp.al.Scorer;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.stats.CategoricalDistribution;
import edu.byu.nlp.stats.ConditionalCategoricalDistribution;

/**
 * @author rah67
 *
 */
public class QBUScorer implements Scorer {

    public static interface UncertaintyCalculator {
        double uncertaintyOf(CategoricalDistribution dist);
    }
    
    public static class Entropy implements UncertaintyCalculator {

        /** {@inheritDoc} */
        @Override
        public double uncertaintyOf(CategoricalDistribution dist) {
            return dist.entropy();
        }
    }
    
    public static class LeastConfident implements UncertaintyCalculator {

        /** {@inheritDoc} */
        @Override
        public double uncertaintyOf(CategoricalDistribution dist) {
            return 1.0 - Math.exp(dist.logMax());
        }
        
    }
    
    public static class Nlmp implements UncertaintyCalculator {

        /** {@inheritDoc} */
        @Override
        public double uncertaintyOf(CategoricalDistribution dist) {
            return -dist.logMax();
        }
        
    }
    
    private final UncertaintyCalculator uncertaintyCalculator;
    private ConditionalCategoricalDistribution<SparseFeatureVector> condDist;
    private long age;
    
    public QBUScorer(UncertaintyCalculator uncertaintyCalculator,
                     ConditionalCategoricalDistribution<SparseFeatureVector> condDist) {
        this.uncertaintyCalculator = uncertaintyCalculator;
        this.condDist = condDist;
    }

    public synchronized void setCondDist(ConditionalCategoricalDistribution<SparseFeatureVector> condDist, long age) {
        this.condDist = condDist;
        this.age = age;
    }


    /** {@inheritDoc} */
    @Override
    public Score score(FlatInstance instance) {
//      CategoricalDistribution dist = condDist.given(instance.asFeatureVector());
//      return new Score(uncertaintyCalculator.uncertaintyOf(dist), age);
      // FIXME (pfelt): see not in LabelUncertaintyScorer
      return null;
    }
}
