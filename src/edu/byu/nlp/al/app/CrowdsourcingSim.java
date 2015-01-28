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
package edu.byu.nlp.al.app;

import java.io.FileNotFoundException;

import org.apache.commons.vfs2.FileSystemException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.byu.nlp.util.jargparser.annotations.Option;

/**
 * @author rah67
 * 
 */
public class CrowdsourcingSim {

  private static final Logger logger = LoggerFactory.getLogger(CrowdsourcingSim.class);

  @Option(help = "Accuracy levels of annotators. The first one assumed arbiter for ab1 and ab2")
  private static final double[] accuracies = new double[] { 0.90, 0.80, 0.70 };

  @Option
  private static final double[] meanTimeInSecs = new double[] { 25, 30, 35 };

  @Option
  private static double intercept = 3.0 /* secs */;

  @Option
  private static double sdFactor = 10.0 / 2.0; // mean is multiplied by this
                                               // number to obtain sd.

  @Option
  private static int k = 1; // Used by grr/ab

  private enum InstanceManagers {
    grr, ab, single
  }; // ab1, ab2, entropy, least_conf, nlmp, ve };

  @Option
  private static InstanceManagers instanceManager = InstanceManagers.grr;

  private static enum LabelEstimators {
    none, entropy, tail_prob
  };

  @Option
  private static LabelEstimators labelEstimator = LabelEstimators.none;

  private static enum ModelEstimators {
    none, entropy, least_conf, nlmp, ve
  };

  @Option
  private static ModelEstimators modelEstimator = ModelEstimators.none;

  @Option
  private static boolean estimateCost = false;

  @Option(help = "base directory of the documents")
  private static String basedir = "20_newsgroups";

  @Option
  private static String dataset = "tiny_set";

  @Option
  private static String split = "all";

  @Option
  private static int minFeaturesToKeepPerDocument = 10;

  @Option
  private static long seed = -1;

  @Option
  private static String annotationsFile = null;

  @Option
  private static String resultsFile = null;

  @Option
  private static double splitPercent = 0.85;

  @Option
  private static int committeeSize = 5;

  @Option
  private static double luTailProbPrior = 1.0;

  @Option
  private static int luTailProbNumSamples = 1000;

  public static void main(String[] args) throws FileNotFoundException, FileSystemException {
//    args = new ArgumentParser(MultiAnnSim.class).parseArgs(args).getPositionalArgs();
//
//    if (seed == -1) {
//      seed = System.nanoTime();
//    }
//
//    // TODO(rhaertel): reconcile the two dataset classes.
//    Dataset data = readData(createRandomGenerator(seed));
//    List<Dataset> partitions = data.split(splitPercent);
//
//    Set<Long> arbiters;
//    LabelChooser chooser;
//    if (instanceManager == InstanceManagers.ab) {
//      arbiters = ImmutableSet.of(0L);
//      chooser = new ArbiterVote(arbiters, createRandomGenerator(seed));
//    } else {
//      arbiters = Collections.emptySet();
//      chooser = new MajorityVote(createRandomGenerator(seed));
//    }
//
//    DatasetBuilder datasetBuilder = new DatasetBuilder(chooser);
//
//    InstanceManager<Integer, SparseFeatureVector> instanceManager =
//        createInstanceManager(partitions.get(0), createRandomGenerator(seed), arbiters,
//                              datasetBuilder);
//    PrintWriter annotationsOut = annotationsFile == null ? null : new PrintWriter(annotationsFile);
//    AnnotationServer<Integer, SparseFeatureVector> server =
//        new DefaultAnnotationServer<Integer, SparseFeatureVector>(instanceManager, 1,
//                                                                  TimeUnit.SECONDS, annotationsOut);
//
//    SingleLabelLabeler labeler =
//        new SingleLabelLabeler(new NaiveBayesLearner(), datasetBuilder, partitions.get(1),
//                               accuracies.length);
//    TransformingFutureIterator<Collection<DatasetInstance>, Predictions> trainer =
//        TransformingFutureIterator.from(instanceManager.newInstanceFutureIterator(),
//                                        new DataToPredictions(labeler));
//
//    Map<Long, AnnotatorInfo> annotatorInfos = createAnnotatorInfos();
//    PrintWriter resultsOut = resultsFile == null ? null : new PrintWriter(resultsFile);
//    Evaluator<Integer, SparseFeatureVector> evaluator =
//        new Evaluator<Integer, SparseFeatureVector>(trainer, data, resultsOut, annotatorInfos);
//    Thread evaluatorThread = new Thread(evaluator, "Evaluator");
//    evaluatorThread.start();
//
//    List<Callable<Void>> tirelessAnnotators = createAnnotators(data, server);
//    runSimulation(tirelessAnnotators);
  }

//  private static Map<Long, AnnotatorInfo> createAnnotatorInfos() {
//    Map<Long, AnnotatorInfo> annotatorInfos = Maps.newHashMap();
//    for (int i = 0; i < accuracies.length; i++) {
//      annotatorInfos.put((long) i, new AnnotatorInfo(accuracies[i]));
//    }
//    return annotatorInfos;
//  }
//
//  private static class AnnotatorInfo {
//    private final double accuracy;
//
//    public AnnotatorInfo(double accuracy) {
//      this.accuracy = accuracy;
//    }
//
//    public double getAccuracy() {
//      return accuracy;
//    }
//
//    public double getHourlyRate() {
//      // return -3.0/(accuracy - 1.0);
//      return 10.0;
//    }
//  }
//
//  private static InstanceManager<Integer, SparseFeatureVector> createInstanceManager(Dataset data,
//                                                                                     RandomGenerator rnd,
//                                                                                     Set<Long> arbiters,
//                                                                                     DatasetBuilder datasetBuilder) {
//    switch (instanceManager) {
//    case grr:
//      return GeneralizedRoundRobinInstanceManager.newManager(k, data.labeledData(), rnd);
//    case ab:
//      if (arbiters.isEmpty()) {
//        throw new IllegalStateException();
//      }
//      return ABArbiterInstanceManager.newManager(data.labeledData(), k == 1, arbiters);
//    // (pfelt) "single" here doesn't imply that no instances will be annotated redundantly
//    // but rather that instances will be selected for annotation one at a time 
//    // (grr and ab in contrast get multiple annotations for an instance simultaneously) 
//    case single:
//      InstanceManagerBuilder builder = null;
//      UniformClassifier initialModel = new UniformClassifier(data.getNumLabels(), createRnd(rnd));
//      switch (modelEstimator) {
//      case ve:
//        QBCScorer.DisagreementCalculator disCalc = new QBCScorer.VoteEntropy(data.getNumLabels());
//        builder = new QBCBuilder(disCalc, initialModel, datasetBuilder, committeeSize, rnd);
//        break;
//      case entropy:
//        builder = new QBUBuilder(new QBUScorer.Entropy(), initialModel, datasetBuilder);
//        break;
//      case least_conf:
//        builder = new QBUBuilder(new QBUScorer.LeastConfident(), initialModel, datasetBuilder);
//        break;
//      case nlmp:
//        builder = new QBUBuilder(new QBUScorer.Nlmp(), initialModel, datasetBuilder);
//        break;
//      case none:
//        break;
//      default:
//        throw new IllegalArgumentException("I do not know how to create the " + modelEstimator
//            + " model estimator");
//      }
//      LabelUncertaintyScorer luScorer = null;
//      switch (labelEstimator) {
//      case entropy:
//        luScorer =
//            new LabelUncertaintyScorer(new LabelUncertaintyScorer.Entropy(), data.getNumLabels());
//        break;
//      case tail_prob:
//        MonteCarloTailProb uncCalc =
//            new MonteCarloTailProb(luTailProbPrior, luTailProbNumSamples, createRnd(rnd));
//        luScorer = new LabelUncertaintyScorer(uncCalc, data.getNumLabels());
//        break;
//      case none:
//        break;
//      default:
//        throw new IllegalArgumentException("I do not know how to create the " + labelEstimator
//            + " label estimator");
//      }
//      return new SingleQueueInstanceManagerFactory(builder, luScorer).newInstanceManager(data.labeledData());
//    default:
//      throw new IllegalArgumentException("I do not know how to create the " + instanceManager
//          + " instance manager");
//    }
//  }
//
//  private static RandomGenerator createRnd(RandomGenerator masterRnd) {
//    return new MersenneTwister(masterRnd.nextLong());
//  }
//
//  /**
//   * (pfelt) This class is in charge of taking a model scorer and a label 
//   * scorer, and combining them into either:
//   * LU (if builder==null && luScorer!=null)
//   * LMU (if builder!=null && luScorer!=null)
//   * MU (if builder!=null && luScorer==null)  
//   * 
//   * Each of these scorers may optionally be combined with a cost model 
//   */
//  // Factory pattern; this class also acts as the Director in the Builder
//  // pattern.
//  private static class SingleQueueInstanceManagerFactory {
//    private final InstanceManagerBuilder builder;
//    private final LabelUncertaintyScorer luScorer;
//
//    public SingleQueueInstanceManagerFactory(InstanceManagerBuilder builder,
//                                             LabelUncertaintyScorer luScorer) {
//      this.builder = builder;
//      this.luScorer = luScorer;
//    }
//
//    public InstanceManager<Integer, SparseFeatureVector> newInstanceManager(Iterable<DatasetInstance> instances) {
//      Scorer<Integer, SparseFeatureVector> scorer = null;
//      if (builder != null) {
//        builder.newScorer();
//        scorer = builder.getScorer();
//        if (luScorer != null) {
//          scorer = new LabelAndModelScorer(scorer, luScorer);
//        }
//      } else if (luScorer != null) {
//        scorer = luScorer;
//      } else {
//        throw new IllegalStateException();
//      }
//
//      Preconditions.checkState(scorer != null);
//
//      TimeScorer costScorer = null;
//      if (estimateCost) {
//        costScorer = new TimeScorer(new TimeModel(10, 10), 0);
//        scorer = new ROIScorer<Integer, SparseFeatureVector>(scorer, costScorer);
//      }
//      SharedQueueInstanceManager<Integer, SparseFeatureVector> im =
//          SharedQueueInstanceManager.newProvider(instances, scorer);
//      builder.createAndStartUpdaterThreads(im);
//      if (costScorer != null) {
//        new Thread(new CostModelUpdater(im.newInstanceFutureIterator(), costScorer),
//                   "CostModelUpdater").start();
//      }
//      return im;
//    }
//  }
//
//  private static interface InstanceManagerBuilder {
//    void newScorer();
//
//    Scorer<Integer, SparseFeatureVector> getScorer();
//
//    void createAndStartUpdaterThreads(InstanceManager<Integer, SparseFeatureVector> im);
//  }
//
//  private static class QBUBuilder implements InstanceManagerBuilder {
//    private final QBUScorer.UncertaintyCalculator uncCalc;
//    private final ConditionalCategoricalDistribution<SparseFeatureVector> initialModel;
//    private final DatasetBuilder datasetBuilder;
//
//    private QBUScorer qbuScorer;
//
//    public QBUBuilder(UncertaintyCalculator uncCalc,
//                      ConditionalCategoricalDistribution<SparseFeatureVector> initialModel,
//                      DatasetBuilder datasetBuilder) {
//      this.uncCalc = uncCalc;
//      this.initialModel = initialModel;
//      this.datasetBuilder = datasetBuilder;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void newScorer() {
//      this.qbuScorer = new QBUScorer(uncCalc, initialModel);
//    }
//
//    @Override
//    public Scorer<Integer, SparseFeatureVector> getScorer() {
//      return qbuScorer;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void createAndStartUpdaterThreads(InstanceManager<Integer, SparseFeatureVector> im) {
//      Preconditions.checkState(qbuScorer != null, "qbuScorer is null; did you call newScorer()?");
//      new Thread(new QBUScorerUpdater(im.newInstanceFutureIterator(), new NaiveBayesLearner(),
//                                      datasetBuilder, qbuScorer), "ScorerUpdater").start();
//    }
//  }
//
//  private static class QBCBuilder implements InstanceManagerBuilder {
//    private final QBCScorer.DisagreementCalculator disCalc;
//    private final ConditionalCategoricalDistribution<SparseFeatureVector> initialModel;
//    private final DatasetBuilder datasetBuilder;
//    private final int committeeSize;
//    private final RandomGenerator rnd;
//
//    private QBCScorer qbcScorer;
//
//    public QBCBuilder(DisagreementCalculator disCalc,
//                      ConditionalCategoricalDistribution<SparseFeatureVector> initialModel,
//                      DatasetBuilder datasetBuilder, int committeeSize, RandomGenerator rnd) {
//      this.disCalc = disCalc;
//      this.initialModel = initialModel;
//      this.datasetBuilder = datasetBuilder;
//      this.committeeSize = committeeSize;
//      this.rnd = rnd;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void newScorer() {
//      this.qbcScorer = QBCScorer.from(disCalc, Collections.nCopies(committeeSize, initialModel));
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public Scorer<Integer, SparseFeatureVector> getScorer() {
//      return qbcScorer;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void createAndStartUpdaterThreads(InstanceManager<Integer, SparseFeatureVector> im) {
//      Preconditions.checkState(qbcScorer != null, "qbuScorer is null; did you call newScorer()?");
//      for (int i = 0; i < committeeSize; i++) {
//        // Note (rhaertel): is there a way to avoid this two-way dependency?
//        FutureIterator<Collection<DatasetInstance>> it =
//            im.newInstanceFutureIterator();
//        RandomGenerator newRnd = new MersenneTwister(rnd.nextLong());
//        Runnable updater =
//            new QBCScorerUpdater(it, new NaiveBayesLearner(), datasetBuilder, qbcScorer, i, newRnd);
//        new Thread(updater, String.format("ScorerUpdater-%02d", i)).start();
//      }
//    }
//  }
//
//  private static class TimeScorer implements Scorer<Integer, SparseFeatureVector> {
//    private TimeModel model;
//    private long age;
//
//    public TimeScorer(TimeModel model, long age) {
//      this.model = model;
//      this.age = age;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public Score score(DatasetInstance instance) {
//      return new Score(model.timeFor(instance.getData()), age);
//    }
//
//    public void setModel(TimeModel model, long age) {
//      this.model = model;
//      this.age = age;
//    }
//  }
//
//  private static class CostModelUpdater implements Runnable {
//    private final FutureIterator<Collection<DatasetInstance>> it;
//    private final TimeScorer scorer;
//
//    public CostModelUpdater(FutureIterator<Collection<DatasetInstance>> it,
//                            TimeScorer scorer) {
//      this.it = it;
//      this.scorer = scorer;
//    }
//
//    @Override
//    public void run() {
//      while (!Thread.interrupted()) {
//        try {
//          Collection<DatasetInstance> data = it.next();
//          SimpleRegression reg = new SimpleRegression();
//          // TODO(rhaertel): consider creating new datatype with numAnnotations
//          // and iterator of instances
//          for (DatasetInstance instance : data) {
//            for (TimedAnnotation<Integer> ta : instance.getAnnotations().values()) {
//              long totalNanos =
//                  ta.getAnnotationTime().getDurationNanos()
//                      + ta.getAnnotationTime().getDurationNanos();
//              double size = Math.max(1.0, instance.getData().sum());
//              reg.addData(Math.log(size), (double) totalNanos);
//            }
//          }
//          if (reg.getN() < 3) {
//            continue;
//          }
//          double[] params = reg.regress().getParameterEstimates();
//          logger.info(String.format("Time Model = %f, %f", params[0], params[1]));
//          scorer.setModel(new TimeModel(params[0], params[1]), reg.getN());
//        } catch (InterruptedException e) {
//          Thread.currentThread().interrupt();
//        }
//      }
//    }
//  }
//
//  private static class QBUScorerUpdater implements Runnable {
//    private final FutureIterator<Collection<DatasetInstance>> it;
//    private final NaiveBayesLearner learner;
//    private final DatasetBuilder builder;
//    private final QBUScorer scorer;
//
//    public QBUScorerUpdater(FutureIterator<Collection<DatasetInstance>> it,
//                            NaiveBayesLearner learner, DatasetBuilder builder, QBUScorer scorer) {
//      this.it = it;
//      this.learner = learner;
//      this.builder = builder;
//      this.scorer = scorer;
//    }
//
//    @Override
//    public void run() {
//      while (!Thread.interrupted()) {
//        try {
//          Collection<DatasetInstance> instances = it.next();
//          long numAnnotations = 0;
//          for (DatasetInstance instance : instances) {
//            numAnnotations += instance.getAnnotations().size();
//          }
//          edu.byu.nlp.al.classify2.Dataset dataset = builder.buildDataset(instances, null);
//          // TODO(rhaertel): promote to Collection to avoid size() overhead
//          long age = numAnnotations;
//          scorer.setCondDist(learner.learnFrom(dataset), age);
//        } catch (InterruptedException e) {
//          Thread.currentThread().interrupt();
//        }
//      }
//    }
//  }
//
//  private static class QBCScorerUpdater implements Runnable {
//
//    private final FutureIterator<Collection<DatasetInstance>> it;
//    private final NaiveBayesLearner learner;
//    private final DatasetBuilder builder;
//    private final QBCScorer scorer;
//    private final int index;
//    private final RandomGenerator rnd;
//
//    public QBCScorerUpdater(FutureIterator<Collection<DatasetInstance>> it,
//                            NaiveBayesLearner learner, DatasetBuilder builder, QBCScorer scorer,
//                            int index, RandomGenerator rnd) {
//      this.it = it;
//      this.learner = learner;
//      this.builder = builder;
//      this.scorer = scorer;
//      this.index = index;
//      this.rnd = rnd;
//    }
//
//    @Override
//    public void run() {
//      while (!Thread.interrupted()) {
//        try {
//          Collection<DatasetInstance> instances = it.next();
//          long numAnnotations = 0;
//          for (DatasetInstance instance : instances) {
//            numAnnotations += instance.getAnnotations().size();
//          }
//          edu.byu.nlp.al.classify2.Dataset bootstrapSample =
//              builder.buildDataset(instances, null).bootstrapSample(rnd);
//          long age = numAnnotations;
//          scorer.setDist(index, learner.learnFrom(bootstrapSample), age);
//        } catch (InterruptedException e) {
//          Thread.currentThread().interrupt();
//        }
//      }
//    }
//  }
//
//  /**
//   * All costs are in seconds.
//   */
//  private static class Cost {
//    private final double annotationSecs;
//    private final double waitSecs;
//    private final double annotationCost;
//    private final double waitCost;
//
//    public Cost(double annotationSecs, double waitSecs, double annotationCost, double waitCost) {
//      this.annotationSecs = annotationSecs;
//      this.waitSecs = waitSecs;
//      this.annotationCost = annotationCost;
//      this.waitCost = waitCost;
//    }
//
//    public double getAnnotationSecs() {
//      return annotationSecs;
//    }
//
//    public double getWaitSecs() {
//      return waitSecs;
//    }
//
//    public double getAnnotationCost() {
//      return annotationCost;
//    }
//
//    public double getWaitCost() {
//      return waitCost;
//    }
//
//    public double getTotalCost() {
//      return annotationCost + waitCost;
//    }
//
//    public String toCsv() {
//      return Joiner.on(", ").join(getWaitSecs(), getAnnotationSecs(), getWaitCost(),
//                                  getAnnotationCost(), getTotalCost());
//    }
//  }
//
//
//  public static class Evaluator<L, D> implements Runnable {
//
//    private final FutureIterator<Predictions> it;
//    private final PrintWriter out;
//    private final Map<Long, AnnotatorInfo> annotatorInfos; // FIXME(rhaertel):
//    private Dataset data;                                  // annotatorInfos and data
//                                                           // only needed for
//                                                           // calculator, would
//                                                           // be better to
//                                                           // inject the
//                                                           // calculator
//
//    public Evaluator(FutureIterator<Predictions> it, Dataset data, @Nullable PrintWriter out,
//                     Map<Long, AnnotatorInfo> annotatorInfos) {
//      Preconditions.checkNotNull(it);
//      this.it = it;
//      this.out = out;
//      this.annotatorInfos = annotatorInfos;
//      this.data=data;
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public void run() {
//      try {
//        AnnotationsCounter annotationsCounter = new AnnotationsCounter();
//        CostComputer costComputer = new CostComputer(annotatorInfos);
//        AccuracyComputer accuracyComputer = new AccuracyComputer();
//        AnnotatorAccuracyComputer annAccComputer = new AnnotatorAccuracyComputer(accuracies.length);
//        RmseAnnotatorAccuracyComputer rmseComputer = new RmseAnnotatorAccuracyComputer(accuracies);
//        MachineAccuracyComputer machineAccComputer = new MachineAccuracyComputer();
//        RmseMachineAccuracyVsTestComputer machineRmseComputer = new RmseMachineAccuracyVsTestComputer();
//        if (out != null) {
//          out.println(Joiner.on(", ").join(annotationsCounter.csvHeader(),
//                                           costComputer.csvHeader(), accuracyComputer.csvHeader(),
//                                           annAccComputer.csvHeader(), rmseComputer.csvHeader(),
//                                           machineAccComputer.csvHeader(), machineRmseComputer.csvHeader()));
//        }
//        while (!Thread.currentThread().isInterrupted()) {
//          Predictions predictions = it.next();
//          if (predictions == null) {
//            // FIXME(rhaertel): in some cases, this may cause us to hang.
//            break;
//          }
//          String annotationInfo = annotationsCounter.compute(data);
//          Cost cost = costComputer.compute(predictions);
//          // TODO: do we need to consider unlabeled instances (setting the null label to identify them)?
//          OverallAccuracy overallAccuracy = accuracyComputer.compute(predictions, null); 
//          DoubleArrayCsvAble annAccuracy = annAccComputer.compute(predictions);
//          double rmse = rmseComputer.compute(predictions);
//          double machineAcc = machineAccComputer.compute(predictions);
//          double machineRmse = machineRmseComputer.compute(predictions, null);
//          writeData(annotationInfo, cost, overallAccuracy, annAccuracy, rmse, machineAcc, machineRmse);
//          logger.info(String.format("Num Annotations = %d, Annotation Time = %f, Wait Cost=%f, "
//                                        + "Annotation Cost=%f, Total Cost=%f, Accuracy=%s, RMSE=%f",
//                                    annotationInfo,
//                                    cost.getAnnotationSecs(), cost.getWaitCost(),
//                                    cost.getAnnotationCost(), cost.getTotalCost(), overallAccuracy,
//                                    rmse));
//        }
//      } catch (InterruptedException e) {
//        Thread.currentThread().interrupt();
//      }
//      if (out != null) {
//        out.flush();
//      }
//    }
//
//    private void writeData(String annotationInfo, Cost cost, OverallAccuracy overallAccuracy,
//                           DoubleArrayCsvAble annAccuracy, double rmse, double machineAcc, double machineRmse) {
//      if (out != null) {
//        out.println(Joiner.on(", ").join(annotationInfo, cost.toCsv(), overallAccuracy.toCsv(),
//                                         annAccuracy.toCsv(), rmse, machineAcc, machineRmse));
//        out.flush();
//      }
//    }
//  }
//
//  /**
//   * Decorates one FutureIterator and transforms the result.
//   */
//  private static class TransformingFutureIterator<F, T> implements FutureIterator<T> {
//    private final FutureIterator<F> it;
//    private final Function<F, T> func;
//
//    public TransformingFutureIterator(FutureIterator<F> it, Function<F, T> func) {
//      this.it = it;
//      this.func = func;
//    }
//
//    public static <F, T> TransformingFutureIterator<F, T> from(FutureIterator<F> it,
//                                                               Function<F, T> func) {
//      return new TransformingFutureIterator<F, T>(it, func);
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public T next() throws InterruptedException {
//      F next = it.next();
//      if (next == null) {
//        return null;
//      }
//      return func.apply(next);
//    }
//
//    /** {@inheritDoc} */
//    @Override
//    public T next(long timeout, TimeUnit unit) throws InterruptedException {
//      F next = it.next(timeout, unit);
//      if (next == null) {
//        return null;
//      }
//      return func.apply(next);
//    }
//  }
//
//  private static class DataToPredictions implements
//      Function<Collection<DatasetInstance>, Predictions> {
//    private final DatasetLabeler<Integer, SparseFeatureVector> labeler;
//
//    public DataToPredictions(DatasetLabeler<Integer, SparseFeatureVector> labeler) {
//      this.labeler = labeler;
//    }
//
//    @Override
//    public Predictions apply(Collection<DatasetInstance> data) {
//      return labeler.label(data);
//    }
//  }
//
//  private static List<Callable<Void>> createAnnotators(Dataset data,
//                                                       AnnotationServer<Integer, SparseFeatureVector> server) {
//    final int numAnnotators = accuracies.length;
//    GoldLabelProvider goldLabelProvider =
//        GoldLabelProvider.from(data.labeledData());
//    List<Callable<Void>> tirelessAnnotators = Lists.newArrayListWithCapacity(numAnnotators);
//    for (int a = 0; a < numAnnotators; a++) {
//      // Note(rhaertel): should be safe to share RandomGenerator amongst
//      // LabelErrorFunction and
//      // TimeSimulator.
//      RandomGenerator rnd = createRandomGenerator(seed / (a + 1));
//      TimeSimulator<SparseFeatureVector> timeSimulator =
//          new TextClassificationTimeSimulator(new TimeModel(intercept * 1e9,
//                                                            slope(meanTimeInSecs[a], intercept,
//                                                                  data) * 1e9), sdFactor, rnd);
//      ProbabilisticLabelErrorFunction<Integer> labelErrorFunction =
//          new ProbabilisticLabelErrorFunction<Integer>(
//                                                       new AccuracyDistribution(accuracies[a],
//                                                                                data.getNumLabels()),
//                                                       rnd);
//      FallibleAnnotationProvider<Integer, SparseFeatureVector> labelProvider =
//          new FallibleAnnotationProvider<Integer, SparseFeatureVector>(goldLabelProvider,
//                                                                       labelErrorFunction);
//      AnnotationInfoProvider<Integer, SparseFeatureVector> aip =
//          new SimulatedTimeAnnotationInfoProvider<Integer, SparseFeatureVector>(labelProvider,
//                                                                                timeSimulator);
//      Timer timer;
//      if (a == 0 && (instanceManager == InstanceManagers.ab)) {
//        timer = Timers.zeroTimer();
//      } else {
//        timer = Timers.systemTimer();
//      }
//      tirelessAnnotators.add(new TirelessAnnotator<Integer, SparseFeatureVector>(a, server, aip,
//                                                                                 true, timer));
//    }
//    return tirelessAnnotators;
//  }
//
//  private static double slope(double meanTimeInSecs, double intercept, Dataset data) {
//    // mean = 1 / N * \sum_i (intercept + slope * log(len[i]))
//    // mean = intercept + 1 / N * slope * \sum_i log(len[i])
//    // N * (mean - intercept) / (sum_i log(len[i])) = slope
//    Collection<DatasetInstance> labeledData = data.labeledData();
//    double sumOfLogOfLength = 0.0;
//    for (DatasetInstance instance : labeledData) {
//      double size = Math.max(1.0, instance.getData().sum());
//      sumOfLogOfLength += Math.log(size);
//    }
//    return labeledData.size() * (meanTimeInSecs - intercept) / sumOfLogOfLength;
//  }
//
//  public static RandomGenerator createRandomGenerator(long seed) {
//    return new MersenneTwister(seed);
//  }
//
//  private static Dataset readData(RandomGenerator rnd) throws FileSystemException {
//    Function<List<String>, List<String>> tokenTransform = null; // FIXME
//    DocumentDatasetBuilder newsgroups =
//        new DocumentDatasetBuilder(basedir, dataset, split, new EmailHeaderStripper(),
//            TokenizerPipes.McCallumAndNigam(), tokenTransform , new TopNPerDocumentFeatureSelectorFactory<String>(
//                minFeaturesToKeepPerDocument));
//    Dataset data = newsgroups.dataset();
//
//    logger.info("Number of instances = " + data.labeledData().size());
//    logger.info("Number of tokens = " + data.getNumTokens());
//    logger.info("Number of features = " + data.getNumFeatures());
//    logger.info("Number of classes = " + data.getNumLabels());
//
//    data.shuffle(rnd);
//    data = data.copy();
//    return data;
//  }
//
//  private static ThreadFactory annotatorThreadFactory() {
//    final ThreadGroup group = new ThreadGroup("Annotators");
//    return new ThreadFactory() {
//
//      int i = 0;
//
//      @Override
//      public Thread newThread(Runnable r) {
//        return new Thread(group, r, "annotator-" + (i++));
//      }
//    };
//  }
//
//  private static void runSimulation(List<Callable<Void>> tirelessAnnotators) {
//    ExecutorService executor =
//        Executors.newFixedThreadPool(tirelessAnnotators.size(), annotatorThreadFactory());
//    try {
//      List<Future<Void>> futures = executor.invokeAll(tirelessAnnotators);
//      executor.shutdown();
//      for (Future<Void> future : futures) {
//        // At this point the annotators are done, but this allows us to retrieve
//        // any exceptions that might have
//        // been thrown.
//        try {
//          future.get();
//        } catch (ExecutionException e) {
//          e.printStackTrace();
//        } catch (InterruptedException ie) {
//          ie.printStackTrace();
//        }
//      }
//    } catch (InterruptedException ie) {
//      ie.printStackTrace();
//    }
//  }
}
