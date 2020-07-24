package org.tensorflow.lite.examples.speech;

import androidx.core.util.Pair;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public class RecognizeCommandsTest {
  Deque<Pair<Long, float[]>> workingDeque = new ArrayDeque<Pair<Long, float[]>>();
  float[] ephemeralResult = new float[3];

  @Test
  public void recogniseCommands_RefreshPreviousResultsBuffer_CorrectlyAveragesPredictions() {

    for (int i = 0; i < 3; i++) {
      Long dummyTime = (long) (i * 10);
      // fill dummy data to simulate the updating of predictions by tfLite.run
      Arrays.fill(ephemeralResult, (float) i);

      workingDeque =
          RecognizeCommands.RefreshPreviousResultsBuffer(
              workingDeque, 100L, ephemeralResult, dummyTime);
    }

    // Assert that the final value of the result in the Deque did not overwrite the first value
    // added
    Assert.assertFalse(workingDeque.getFirst().second.equals(workingDeque.getLast().second));
  }
}
