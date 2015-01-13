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

import edu.byu.nlp.al.Scorer;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;


/**
 * @author rah67
 *
 */
public class LabelAndModelScorer implements Scorer<SparseFeatureVector,Integer> {
    private final Scorer<SparseFeatureVector,Integer> modelScorer;
    private final LabelUncertaintyScorer luScorer;

    public LabelAndModelScorer(Scorer<SparseFeatureVector,Integer> modelScorer, LabelUncertaintyScorer luScorer) {
        this.modelScorer = modelScorer;
        this.luScorer = luScorer;
    }

    /** {@inheritDoc} */
    @Override
    public edu.byu.nlp.al.Scorer.Score score(FlatInstance<SparseFeatureVector, Integer> instance) {
        Score qbuScore = modelScorer.score(instance);
        Score luScore = luScorer.score(instance);
        return new Score(Math.sqrt(qbuScore.getScore() * luScore.getScore()), qbuScore.getBirthDate());
    }

}
