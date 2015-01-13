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
package edu.byu.nlp.al.simulation;

import com.google.common.base.Function;

import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.data.types.SparseFeatureVector;

/**
 * Provides annotation as if from a fallible annotator.
 * 
 * @author rah67
 *
 */
public class FallibleAnnotationProvider<D,L> implements LabelProvider<D,L> {

	private final GoldLabelProvider<D,L> goldLabelProvider;
	private final Function<L, L> labelErrorFunction;
	
	public FallibleAnnotationProvider(GoldLabelProvider<D,L> goldLabelProvider, Function<L, L> labelErrorFunction) {
		this.goldLabelProvider = goldLabelProvider;
		this.labelErrorFunction = labelErrorFunction;
	}

	/** {@inheritDoc} */
	@Override
  public L labelFor(String source, D datum) {
		return labelErrorFunction.apply(goldLabelProvider.labelFor(source, datum));
	}
	
	public static FallibleAnnotationProvider<SparseFeatureVector,Integer> from(
	    GoldLabelProvider<SparseFeatureVector,Integer> goldLabelProvider, Function<Integer, Integer> labelErrorFunction){
	  
	  return new FallibleAnnotationProvider<SparseFeatureVector,Integer>(goldLabelProvider, labelErrorFunction);
	}

}
