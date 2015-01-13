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

import java.rmi.RemoteException;

/**
 * Remote interface to an AnnotationServer, which essentially handles the management of requests for instances,
 * presumably delegating the real work to an InstanceManager.
 * 
 * @author rah67
 *
 */
public interface AnnotationServer<D, L> {
	void cancelInstanceRequest(long requestId) throws RemoteException;
	RequestInstanceResponse<D> requestInstanceFor(long annotatorId) throws RemoteException;
	void storeAnnotation(AnnotationInfo<L> annotationInfo) throws RemoteException;
	void shutdown();
}
