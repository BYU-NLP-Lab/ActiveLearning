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
package edu.byu.nlp.al;

import edu.byu.nlp.data.FlatInstance;



/**
 * @author rah67
 *
 */
public class ROIScorer<D,L> implements Scorer<D,L> {

    private final Scorer<D,L> benefitScorer;
    private final Scorer<D,L> costScorer;
    
    public ROIScorer(Scorer<D,L> benefitScorer, Scorer<D,L> costScorer) {
        this.benefitScorer = benefitScorer;
        this.costScorer = costScorer;
    }

    /** {@inheritDoc} */
    @Override
    public Score score(FlatInstance<D, L> instance) {
        Score benefit = benefitScorer.score(instance);
        Score cost = costScorer.score(instance);
        return new Score(benefit.getScore() / cost.getScore(), Math.min(benefit.getBirthDate(), cost.getBirthDate()));
    }
}
