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

import static edu.byu.nlp.stats.DoubleArrayCategoricalDistribution.newDistributionFromProbs;
import static java.lang.Math.log;
import static org.fest.assertions.Delta.delta;

import java.util.List;

import org.fest.assertions.Assertions;
import org.junit.Test;

import com.google.common.collect.Lists;

import edu.byu.nlp.al.classify.QBCScorer.VoteEntropy;
import edu.byu.nlp.stats.CategoricalDistribution;

/**
 * @author rah67
 *
 */
public class QBCScorerTest {

    @Test
    public void testVoteEntropy() {
        VoteEntropy ve = new QBCScorer.VoteEntropy(4);
        List<CategoricalDistribution> labelDists = Lists.newArrayList();
        labelDists.add(newDistributionFromProbs(new double[] {0.1, 0.7, 0.2}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.2, 0.6, 0.2}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.1, 0.8, 0.1}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.3, 0.2, 0.5}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.4, 0.1, 0.5}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.1, 0.2, 0.7}, false, 1e-14));
        labelDists.add(newDistributionFromProbs(new double[] {0.7, 0.1, 0.2}, false, 1e-14));
        
        // 3/7, 3/7, 1/7
        final double expected = -3./7. * log(3./7.) - 3./7. * log(3./7.) - 1./7. * log(1./7.); 
        Assertions.assertThat(ve.disagreementOf(labelDists)).isEqualTo(expected, delta(1e-10));
    }

}
