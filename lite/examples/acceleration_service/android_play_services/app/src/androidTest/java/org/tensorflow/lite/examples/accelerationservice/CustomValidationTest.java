/*
 * Copyright 2023 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.accelerationservice;

import static com.google.common.truth.Truth.assertThat;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.TaskCompletionSource;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.acceleration.AccelerationConfig;
import com.google.android.gms.tflite.acceleration.CpuAccelerationConfig;
import com.google.android.gms.tflite.acceleration.CustomValidationConfig;
import com.google.android.gms.tflite.acceleration.CustomValidationConfig.AccuracyValidator;
import com.google.android.gms.tflite.acceleration.ValidationConfig;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModel;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModelFactory;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModelFactory.ModelType;
import org.tensorflow.lite.examples.accelerationservice.validator.MeanSquaredErrorValidator;

/**
 * Instrumented test, which will execute on an Android device. The test will run the interpreter
 * using CPU config, and then check if the correct inference output is produced.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class CustomValidationTest {

  private final AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder().build();
  private final Executor executor = Executors.newSingleThreadExecutor();

  private AssetModelFactory assetModelFactory;
  private AccuracyValidator validator;

  @Rule
  public ActivityScenarioRule<MainActivity> scenarioRule =
      new ActivityScenarioRule<>(MainActivity.class);

  @Before
  public void setUp() throws ExecutionException, InterruptedException {
    Context context = ApplicationProvider.getApplicationContext();
    Logger logger = new NoopLogger();
    validator = new MeanSquaredErrorValidator(logger, MainActivity.MSE_THRESHOLD);
    assetModelFactory =
        Tasks.await(
            getMainActivity()
                .onSuccessTask(
                    activity -> Tasks.forResult(new AssetModelFactory(context, executor, logger))));
  }

  @Test
  public void cpuCustomValidationOnPlainAdditionModel_succeeds()
      throws ExecutionException, InterruptedException {
    AssetModel assetModel = Tasks.await(assetModelFactory.load(ModelType.PLAIN_ADDITION));
    assertThat(assetModel.getModel()).isNotNull();
    assertThat(Tasks.await(runScenario(assetModel))).isTrue();
  }

  @Test
  public void cpuCustomValidationOnMobileNetV1Model_succeeds()
      throws ExecutionException, InterruptedException {
    AssetModel assetModel = Tasks.await(assetModelFactory.load(ModelType.MOBILENET_V1));
    assertThat(assetModel.getModel()).isNotNull();
    assertThat(Tasks.await(runScenario(assetModel))).isTrue();
  }

  private Task<Boolean> runScenario(AssetModel assetModel) {
    return getMainActivity()
        .onSuccessTask(
            activity -> {
              ValidationConfig validationConfig =
                  new CustomValidationConfig.Builder()
                      .setGoldenInputs(assetModel.getInputs())
                      .setAccuracyValidator(validator)
                      .setBatchSize(assetModel.getBatchSize())
                      .build();
              return activity.runScenario(
                  executor, assetModel, accelerationConfig, validationConfig);
            });
  }

  private Task<MainActivity> getMainActivity() {
    TaskCompletionSource<MainActivity> taskCompletionSource = new TaskCompletionSource<>();
    scenarioRule.getScenario().onActivity(taskCompletionSource::setResult);
    return taskCompletionSource.getTask();
  }
}
