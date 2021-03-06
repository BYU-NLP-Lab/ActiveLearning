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
package edu.byu.nlp.al;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.fest.assertions.Assertions;
import org.junit.Test;

import com.google.common.collect.Lists;

import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.Counters;
import edu.byu.nlp.util.Iterables2;

/**
 * @author plf1
 *
 */
public class RandomRoundRobinQueueTest {

  @Test
  public void testUnlimited(){
    List<String> source = Lists.newArrayList("a","b","c");
    RandomRoundRobinQueue<String> q = RandomRoundRobinQueue.from(source, 3, new MersenneTwister(1));
    
    ////// first pass
    
    // make sure each item appears exactly 3 times in 1st run.
    Counter<String> itemCount = Counters.count(Iterables2.subInterval(q, 0, 9));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(3*3);
    
    // make sure the three, second, and third groups of three are the same (testing functionality of k)
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 3, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 6, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);

    ////// second pass
    
    // make sure each item appears exactly 3 times in each run.
    itemCount = Counters.count(Iterables2.subInterval(q, 9, 9));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(3*3);

    // make sure the three, second, and third groups of three are the same (testing functionality of k)
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 3, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 6, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);

    ////// 225 passes
    
    // make sure each item appears exactly 3 times in each run.
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 9*225));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3*225);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3*225);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3*225);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(9*225);
    
  }
  
  @Test
  public void testLimited(){
    int numPasses = 5;
    
    List<String> source = Lists.newArrayList("a","b","c");
    RandomRoundRobinQueue<String> q = RandomRoundRobinQueue.from(source, 3, numPasses, new MersenneTwister(1));
    
    ////// first pass
    
    // make sure each item appears exactly 3 times in 1st run.
    Counter<String> itemCount = Counters.count(Iterables2.subInterval(q, 0, 9));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(3*3);
    
    // make sure the three, second, and third groups of three are the same (testing functionality of k)
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 3, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 6, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);

    ////// second pass
    
    // make sure each item appears exactly 3 times in each run.
    itemCount = Counters.count(Iterables2.subInterval(q, 9, 9));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(3*3);

    // make sure the three, second, and third groups of three are the same (testing functionality of k)
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 3, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);
    itemCount = Counters.count(Iterables2.subInterval(q, 6, 3));
    Assertions.assertThat(itemCount.numEntries()).isEqualTo(1);

    ////// all 5 passes
    
    // make sure each item appears exactly 3 times in each run.
    itemCount = Counters.count(Iterables2.subInterval(q, 0, 9*numPasses));
    Assertions.assertThat(itemCount.getCount("a")).isEqualTo(3*numPasses);
    Assertions.assertThat(itemCount.getCount("b")).isEqualTo(3*numPasses);
    Assertions.assertThat(itemCount.getCount("c")).isEqualTo(3*numPasses);
    Assertions.assertThat(itemCount.totalCount()).isEqualTo(9*numPasses);
    
    // ensure it stops after numPasses
    ArrayList<String> qList = Lists.newArrayList(q);
    Assertions.assertThat(qList.size()).isEqualTo(9*numPasses);
    
  }
  
//  @Test
//  public void visusalTest(){
//
//    List<String> source = Lists.newArrayList("a","b","c");
//    RandomRoundRobinQueue<String> q = RandomRoundRobinQueue.from(source, 3, new MersenneTwister(1));
//    System.out.println(Iterables2.subInterval(q, 3, 3));
//    Iterator<String> asd = q.iterator();
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//    System.out.println(asd.next());
//  }
  
}
