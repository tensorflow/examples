/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link TransferLearningModel}. */
@RunWith(AndroidJUnit4.class)
public class TransferLearningModelTest {

  @Test
  public void saveAndLoadShouldPreserveParameters() throws IOException {
    TransferLearningModel model =
        new TransferLearningModel(
            new AssetModelLoader(
                InstrumentationRegistry.getInstrumentation().getContext(), "model"),
            Arrays.asList("1", "2", "3", "4", "5"));

    Path tempFilePath = Files.createTempFile("tflite-tl-test", ".bin");

    model.saveParameters(FileChannel.open(tempFilePath, StandardOpenOption.WRITE));

    byte[] firstContents = Files.readAllBytes(tempFilePath);

    model.close();
    model =
        new TransferLearningModel(
            new AssetModelLoader(
                InstrumentationRegistry.getInstrumentation().getContext(), "model") {
              @Override
              public LiteModelWrapper loadInitializeModel() throws IOException {
                // Fill with ones instead of zeros.
                return new LiteModelWrapper(this.loadMappedFile("softmax_initialize_ones.tflite"));
              }
            },
            Arrays.asList("1", "2", "3", "4", "5"));

    model.loadParameters(FileChannel.open(tempFilePath, StandardOpenOption.READ));
    model.saveParameters(FileChannel.open(tempFilePath, StandardOpenOption.WRITE));

    byte[] secondContents = Files.readAllBytes(tempFilePath);

    Assert.assertArrayEquals(firstContents, secondContents);

    Files.delete(tempFilePath);
  }
}
