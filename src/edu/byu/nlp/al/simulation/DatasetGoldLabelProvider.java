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

import java.util.Map;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Maps;

import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;

/**
 * Uses an identity hash-map to lookup the gold label for an instance.
 * 
 * @author rah67
 *
 */
public class DatasetGoldLabelProvider<D, L> implements LabelProvider<D, L> {

	private final Map<Integer, L> goldLabels;
	
	@VisibleForTesting
	DatasetGoldLabelProvider(Map<Integer, L> goldLabels) {
		this.goldLabels = goldLabels;
	}

	public static DatasetGoldLabelProvider<SparseFeatureVector, Integer> from(Dataset dataset) {
		Map<Integer, Integer> goldLabels = Maps.newHashMap();
		for (DatasetInstance inst: dataset) {
			goldLabels.put(inst.getInfo().getSource(), inst.getLabel());
		}
		return new DatasetGoldLabelProvider<SparseFeatureVector, Integer>(goldLabels);
	}
	
	/** {@inheritDoc} */
	@Override
	  public L labelFor(int source, D datum) {
		return goldLabels.get(datum);
	}

}
