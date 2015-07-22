/**
 * Copyright 2014 Brigham Young University
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

import com.google.common.base.Preconditions;

/**
 * @author rah67
 * @author plf1
 *
 */
public class InstanceForAnnotation<D> {

  private final long requestId;
  private final D instance;
  private final int source;
  
  public InstanceForAnnotation(long requestId, D instance, int source){
    Preconditions.checkNotNull(instance);
    this.requestId=requestId;
    this.instance=instance;
    this.source=source;
  }
  
  public long getRequestId(){
    return requestId;
  }
  
  public D getInstance(){
    return instance;
  }
  
  public int getSource(){
    return source;
  }
  
  /** {@inheritDoc} */
  @Override
  public String toString() {
    return getClass().getName()+" [requestId="+requestId+", instance="+instance+"]";
  }
  
}
