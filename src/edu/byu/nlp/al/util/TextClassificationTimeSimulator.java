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
package edu.byu.nlp.al.util;

import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.al.simulation.TimeSimulator;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.stats.GammaDistribution;

/**
 * TimeSimulator appropriate for use in text classification. Assumes noise is a truncated Normal (disallows negative
 * times) about a TimeModel, which itself is a linear model of the form:
 * 
 * <pre>
 *   time_nanos = intercept + slope * log(length doc)
 * </pre>
 * 
 * @author rah67
 *
 */
public class TextClassificationTimeSimulator implements TimeSimulator<SparseFeatureVector> {
  private TimeModel timeModel;
	private final double sdFactor;
	private final RandomGenerator rnd;
	
  public TextClassificationTimeSimulator(TimeModel timeModel, double sdFactor, RandomGenerator rnd) {
		this.timeModel = timeModel;
		this.sdFactor = sdFactor;
		this.rnd = rnd;
	}

	/** {@inheritDoc} */
	@Override
	public long annotationTimeInNanoSecsFor(SparseFeatureVector vector) {
	  double mean = timeModel.timeFor(vector);
	  double variance = mean * sdFactor;
    // mean = k * scale
    // var = k * scale^2
	  // shape = mean / scale
	  // scale = var / mean
	  // shape = mean^2 / var
	  double shape = mean * mean / variance;
	  double scale = variance / mean;
	  return (long) (GammaDistribution.sample(shape, rnd) * scale);
	}

}
