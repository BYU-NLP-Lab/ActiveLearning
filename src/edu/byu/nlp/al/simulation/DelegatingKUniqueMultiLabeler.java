/**
 * Copyright 2015 Brigham Young University
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

import java.util.Set;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;

import edu.byu.nlp.crowdsourcing.LabelProvider;
import edu.byu.nlp.crowdsourcing.MultiLabelProvider;

/**
 * @author plf1
 *
 * Provides multiple labels simply by querying a delegate label 
 * provider until K unique labels have been returned.
 */
public class DelegatingKUniqueMultiLabeler<D,L> implements MultiLabelProvider<D, L>{

  private LabelProvider<D, L> delegate;
  private int k;

  public static <D,L> DelegatingKUniqueMultiLabeler<D,L> of(LabelProvider<D, L> delegate, int k){
    return new DelegatingKUniqueMultiLabeler<>(delegate, k);
  }
  
  public DelegatingKUniqueMultiLabeler(LabelProvider<D, L> delegate, int k){
    Preconditions.checkNotNull(delegate);
    Preconditions.checkArgument(k>0);
    this.delegate=delegate;
    this.k=k;
  }
  
  /** {@inheritDoc} */
  @Override
  public Iterable<L> labelFor(int source, D datum) {
    Set<L> labels = Sets.newHashSet();
    while (labels.size()<this.k){
      labels.add(delegate.labelFor(source, datum));
    }
    return labels;
  }
  
}
