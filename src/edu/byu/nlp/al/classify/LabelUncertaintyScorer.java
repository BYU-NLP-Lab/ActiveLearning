/**
 * Copyright 2013 Brigham Young University
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

import java.util.Map.Entry;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;

import edu.byu.nlp.al.Scorer;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.math.GammaFunctions;
import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author rah67
 *
 */
public class LabelUncertaintyScorer implements Scorer<SparseFeatureVector,Integer> {

    public static interface UncertaintyCalculator {
        double uncertaintyOf(Counter<Integer> counts);
    }

    public static class Entropy implements UncertaintyCalculator {

        private final double prior = 1.0;
        
        /** {@inheritDoc} */
        @Override
        public double uncertaintyOf(Counter<Integer> counts) {
            double alpha0 = 0.0;
            double entropy = 0.0;
            double[] alpha = new double[counts.numEntries()];
            for (Entry<Integer, Integer> entry : counts.entrySet()) {
                double a = prior + entry.getValue();
                entropy -= (a - 1.0) * Gamma.digamma(a);
                alpha0 += a;
                alpha[entry.getKey()] = a;
            }
            entropy += GammaFunctions.logBeta(alpha) + (alpha0 - counts.numEntries()) * Gamma.digamma(alpha0);
            return entropy;
        }
        
    }
    
    /**
     * Suppose Y_i ~ Cat(\theta) and \theta ~ Dir(\alpha). Wlog, suppose that Y=0 is the most frequent annotation
     * Then, we are interested in the p(argmax(\theta) != 0 | y_1, y_2, ...). We compute this probability via
     * Monte Carlo sampling. 
     */
    public static class MonteCarloTailProb implements UncertaintyCalculator {
        
        private double prior = 1.0;
        private int numSamples = 1000;
        private RandomGenerator rnd;
        
        public MonteCarloTailProb(double prior, int numSamples, RandomGenerator rnd) {
            super();
            this.prior = prior;
            this.numSamples = numSamples;
            this.rnd = rnd;
        }

        /** {@inheritDoc} */
        @Override
        public double uncertaintyOf(Counter<Integer> counts) {
            int argMax = counts.argMax();
            double[] alpha = new double[counts.numEntries()];
            for (Entry<Integer, Integer> entry : counts.entrySet()) {
                alpha[entry.getKey()] = entry.getValue() + prior;
            }
            int count = 0;
            for (int i = 0; i < numSamples; i++) {
                double[] theta = DirichletDistribution.sample(alpha, rnd);
                if (DoubleArrays.argMax(theta) != argMax) {
                    ++count;
                }
            }
            return (double) count / (double) numSamples;
        }
    }
    
    private final int numLabels;
    private final UncertaintyCalculator uncCalc;

    public LabelUncertaintyScorer(UncertaintyCalculator uncCalc, int numLabels) {
        this.uncCalc = uncCalc;
        this.numLabels = numLabels;
    }

    /** {@inheritDoc} */
    @Override
    public edu.byu.nlp.al.Scorer.Score score(FlatInstance<SparseFeatureVector, Integer> instance) {
//        Counter<Integer> counts = 
//                    SparseRealMatrices.countColumns(instance.getAnnotations().getLabelAnnotations(), Datasets.INT_CAST_THRESHOLD);
//        
//        return new Score(uncCalc.uncertaintyOf(counts), 0);
      /*
       *  FIXME (pfelt). The Scorer interface will have to change. It needs access to an aggregated group 
       *  of annotatios, so FlatInsance won't work. And changes here will probably propagate to the rest of the 
       *  AL api. We could either make this API work with annotationinstance Instance interface-type objects,
       *  maintaining generics and compatibility with CCASH, or else with DatasetInstance, eliminating 
       *  generics and making code potentially simpler.  
       */
      return null;
    }


}
