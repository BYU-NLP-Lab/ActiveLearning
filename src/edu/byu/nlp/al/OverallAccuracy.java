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

import com.google.common.base.Joiner;

/**
 * Lazily computes Total Accuracy based on labeled and unlabeled.
 */
public class OverallAccuracy {
	private final Accuracy labeledAccuracy;
	private final Accuracy unlabeledAccuracy;
	private final Accuracy heldoutAccuracy;
	
	public OverallAccuracy(Accuracy labeledAccuracy, Accuracy unlabeledAccuracy, Accuracy heldoutAccuracy) {
		this.labeledAccuracy = labeledAccuracy;
		this.unlabeledAccuracy = unlabeledAccuracy;
		this.heldoutAccuracy = heldoutAccuracy;
	}
	
	public Accuracy getLabeledAccuracy() {
		return labeledAccuracy;
	}
	
	public Accuracy getUnlabeledAccuracy() {
		return unlabeledAccuracy;
	}
	
	public Accuracy getTestAccuracy() {
	    return heldoutAccuracy;
	}
	
	/**
	 * Excludes heldout accuracy.
	 */
	public Accuracy getOverallAccuracy() {
		return new Accuracy(labeledAccuracy.getCorrect() + unlabeledAccuracy.getCorrect(),
				labeledAccuracy.getTotal() + unlabeledAccuracy.getTotal());
	}
	
	@Override
	public String toString() {
		Accuracy overallAccuracy = getOverallAccuracy();
		return String.format("labeled: %d / %d = %f, unlabeled: %d / %d = %f, total: %d / %d = %f, heldout = %d / %d = %f",
				labeledAccuracy.getCorrect(),
				labeledAccuracy.getTotal(),
				labeledAccuracy.getAccuracy(),
				unlabeledAccuracy.getCorrect(),
				unlabeledAccuracy.getTotal(),
				unlabeledAccuracy.getAccuracy(),
				overallAccuracy.getCorrect(),
				overallAccuracy.getTotal(),
				overallAccuracy.getAccuracy(),
				heldoutAccuracy.getCorrect(),
				heldoutAccuracy.getTotal(),
				heldoutAccuracy.getAccuracy());
	}
	
	public String toCsv() {
	    Accuracy overallAccuracy = getOverallAccuracy();
	    return Joiner.on(", ").join(
	            labeledAccuracy.getCorrect(),
	            labeledAccuracy.getTotal(),
                  labeledAccuracy.getAccuracy(),
                  unlabeledAccuracy.getCorrect(),
                  unlabeledAccuracy.getTotal(),
                  unlabeledAccuracy.getAccuracy(),
                  overallAccuracy.getCorrect(),
                  overallAccuracy.getTotal(),
                  overallAccuracy.getAccuracy(),
                  heldoutAccuracy.getCorrect(),
                  heldoutAccuracy.getTotal(),
                  heldoutAccuracy.getAccuracy());
	}
}