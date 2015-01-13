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
package edu.byu.nlp.al;

/**
 * @author rah67
 *
 */
public class RequestInstanceResponse<D> {
	
	public enum Status { NO_INSTANCES, SUCCESS, TIMEOUT, FINISHED };
	
	private final Status status;
	private final InstanceForAnnotation<D> ifa;
	
	private RequestInstanceResponse(Status status, InstanceForAnnotation<D> ifa) {
		this.status = status;
		this.ifa = ifa;
	}

	public Status getStatus() {
		return status;
	}

	public InstanceForAnnotation<D> getInstanceForAnnotation() {
		return ifa;
	}
	
	public static <D> RequestInstanceResponse<D> noInstances() {
		return new RequestInstanceResponse<D>(Status.NO_INSTANCES, null);
	}
	
	public static <D> RequestInstanceResponse<D> success(InstanceForAnnotation<D> ifa) {
		return new RequestInstanceResponse<D>(Status.SUCCESS, ifa);
	}

	public static <D> RequestInstanceResponse<D> timeout() {
		return new RequestInstanceResponse<D>(Status.TIMEOUT, null);
	}
	
	public static <D> RequestInstanceResponse<D> finished() {
	    return new RequestInstanceResponse<D>(Status.FINISHED, null);
	}
}
