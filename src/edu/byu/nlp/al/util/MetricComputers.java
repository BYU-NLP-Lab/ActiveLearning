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
package edu.byu.nlp.al.util;

import java.io.PrintWriter;
import java.util.List;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.eval.AccuracyComputer;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector;
import edu.byu.nlp.util.DoubleArrays;

/**
 * @author pfelt
 *
 */
public class MetricComputers {

//  private static class CostComputer {
//    private Map<Long, AnnotatorInfo> annotatorInfos;
//
//    public CostComputer(Map<Long, AnnotatorInfo> annotatorInfos) {
//      this.annotatorInfos = annotatorInfos;
//    }
//
//    public Cost compute(Predictions predictions) {
//      double cumAnnotationSecs = 0.0;
//      double cumWaitSecs = 0.0;
//      double cumAnnotationCost = 0.0;
//      double cumWaitCost = 0.0;
//      for (Prediction prediction : predictions.labeledPredictions()) {
//        // Type inference is behaving unexpectedly; this is the fix.
//        @SuppressWarnings("unchecked")
//        Multimap<Long, TimedAnnotation<?>> annotations = (Multimap<Long, TimedAnnotation<?>>) (Object) prediction
//            .getInstance().getAnnotations();
//        for (Entry<Long, TimedAnnotation<?>> entry : annotations.entries()) {
//          double hourlyRate = annotatorInfos.get(entry.getKey())
//              .getHourlyRate();
//
//          double annotationSecs = entry.getValue().getAnnotationTime()
//              .getDurationNanos() / 1e9;
//          cumAnnotationSecs += annotationSecs;
//          cumAnnotationCost += annotationSecs * hourlyRate / 3600;
//
//          double waitSecs = entry.getValue().getWaitTime().getDurationNanos() / 1e9;
//          cumWaitSecs += waitSecs;
//          cumWaitCost += waitSecs * hourlyRate / 3600;
//        }
//      }
//      return new Cost(cumAnnotationSecs, cumWaitSecs, cumAnnotationCost,
//          cumWaitCost);
//    }
//
//    public String csvHeader() {
//      return "wait_secs, annotation_secs, wait_cost, annotation_cost, total_cost";
//    }
//  }

  public static class AnnotationsCounter {
    public String compute(Dataset data) {
      int annotationsCount = 0;
      int numInstancesAnnotators = 0;
      for (DatasetInstance inst : data) {
        annotationsCount += inst.getInfo().getNumAnnotations();
        numInstancesAnnotators += inst.getInfo().getNumAnnotations() == 0 ? 0
            : 1;
      }

      return Joiner.on(',').join(
          new String[] { "" + numInstancesAnnotators, "" + annotationsCount,
              "" + data.getInfo().getNumDocumentsWithObservedLabels(), });
    }

    public String csvHeader() {
      return Joiner.on(',').join(
          new String[] { "num_instances_annotated", "num_annotations",
              "num_trusted_labels" });
    }
  }

  public static class DoubleArrayCsvAble {
    private final double[] arr;

    public DoubleArrayCsvAble(double[] arr) {
      this.arr = arr;
    }

    public String toCsv() {
      StringBuilder sb = new StringBuilder();
      if (arr.length > 0) {
        sb.append(arr[0]);
      }
      for (int i = 1; i < arr.length; i++) {
        sb.append(", ");
        sb.append(arr[i]);
      }
      return sb.toString();
    }
  }

  public static class LogJointComputer {

    public double compute(Predictions predictions) {
      return predictions.logJoint();
    }

    public String csvHeader() {
      return "log_joint";
    }
  }

  public static class MachineAccuracyComputer {

    public double compute(Predictions predictions) {
      return predictions.machineAccuracy();
    }

    public String csvHeader() {
      return "machacc";
    }
  }

  public static class PredictionTabulator {
    public static void writeTo(Predictions predictions, PrintWriter writer) {
      // pre-compute largest number of annotations in dataset
      int maxAnnotations = 0;
      for (Prediction pred : predictions.allPredictions()) {
        if (pred.getInstance().getInfo().getNumAnnotations() > maxAnnotations) {
          maxAnnotations = pred.getInstance().getInfo().getNumAnnotations();
        }
      }
      // header
      List<String> annHeader = Lists.newArrayList();
      for (int a = 0; a < maxAnnotations; a++) {
        annHeader.add("" + a);
      }
      writer.println(Joiner.on(',').join(
          Lists.newArrayList("gold", "pred", Joiner.on(',').join(annHeader),
              "source")));
      // body
      for (Prediction pred : predictions.allPredictions()) {
        List<Object> annotations = Lists.newArrayList();
        for (FlatInstance<SparseFeatureVector, Integer> inst : pred
            .getInstance().getAnnotations().getRawLabelAnnotations()) {
          annotations.add(inst.getLabel());
        }
        while (annotations.size() < maxAnnotations) {
          annotations.add("");
        }
        List<?> parts = Lists.newArrayList(pred.getInstance().getLabel(), // gold
            pred.getPredictedLabel() != null ? pred.getPredictedLabel() : "", // predicted
            Joiner.on(',').join(annotations), // annotations
            pred.getInstance().getInfo().getSource()); // source
        writer.println(Joiner.on(',').join(parts));
      }
    }
  }

  public static class AnnotatorAccuracyComputer {

    private final int numAnnotators;

    public AnnotatorAccuracyComputer(int numAnnotators) {
      this.numAnnotators = numAnnotators;
    }

    public DoubleArrayCsvAble compute(Predictions predictions) {
      return new DoubleArrayCsvAble(predictions.annotatorAccuracies());
    }

    public String csvHeader() {
      StringBuilder sb = new StringBuilder();
      if (numAnnotators > 0) {
        sb.append("annacc[");
        sb.append(0);
        sb.append(']');
      }
      for (int i = 1; i < numAnnotators; i++) {
        sb.append(", annacc[");
        sb.append(i);
        sb.append(']');
      }
      return sb.toString();
    }
  }

  public static class RmseAnnotatorAccuracyComputer {
    private final double[] actualAccuracies;

    public RmseAnnotatorAccuracyComputer(double[] accuracies) {
      this.actualAccuracies = accuracies;
    }

    public double compute(Predictions predictions) {
      if (actualAccuracies == null) {
        return -1;
      }
      return DoubleArrays.rmse(predictions.annotatorAccuracies(),
          actualAccuracies);
    }

    public String csvHeader() {
      return "annacc_rmse";
    }
  }

  public static class RmseMachineAccuracyVsTestComputer {
    AccuracyComputer accComputer = new AccuracyComputer();

    public double compute(Predictions predictions, Integer nullLabel) {
      double heldoutAccuracy = accComputer.compute(predictions, nullLabel)
          .getTestAccuracy().getAccuracy();
      return DoubleArrays.rmse(new double[] { predictions.machineAccuracy() },
          new double[] { heldoutAccuracy });
    }

    public String csvHeader() {
      return "machacc_rmse";
    }
  }

}
