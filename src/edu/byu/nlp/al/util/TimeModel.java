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
package edu.byu.nlp.al.util;

import edu.byu.nlp.data.types.SparseFeatureVector;

/**
 * TimeSimulator appropriate for text classification. Linear model of the form:
 * <pre>
 *   time_nanos = intercept + slope * log(length doc)
 * </pre>
 * 
 * @author rah67
 */
public class TimeModel {
    private final double intercept;
    private final double slope;

    public TimeModel(double intercept, double slope) {
        this.intercept = intercept;
        this.slope = slope;
    }

    public double timeFor(SparseFeatureVector vector) {
        double size = Math.max(1.0, vector.sum());
        return intercept + slope * Math.log(size);
    }
}
