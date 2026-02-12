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

import android.content.Context;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.TextView;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.acceleration.AccelerationConfig;
import com.google.android.gms.tflite.acceleration.AccelerationService;
import com.google.android.gms.tflite.acceleration.BenchmarkResult.BenchmarkMetric;
import com.google.android.gms.tflite.acceleration.BenchmarkResult.InferenceOutput;
import com.google.android.gms.tflite.acceleration.CpuAccelerationConfig;
import com.google.android.gms.tflite.acceleration.CustomValidationConfig;
import com.google.android.gms.tflite.acceleration.CustomValidationConfig.AccuracyValidator;
import com.google.android.gms.tflite.acceleration.GpuAccelerationConfig;
import com.google.android.gms.tflite.acceleration.Model;
import com.google.android.gms.tflite.acceleration.ValidatedAccelerationConfigResult;
import com.google.android.gms.tflite.acceleration.ValidationConfig;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
import com.google.android.gms.tflite.java.TfLite;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.InterpreterApi.Options;
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime;
import org.tensorflow.lite.examples.accelerationservice.logger.Logger;
import org.tensorflow.lite.examples.accelerationservice.logger.TextViewLogger;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModel;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModelFactory;
import org.tensorflow.lite.examples.accelerationservice.model.AssetModelFactory.ModelType;
import org.tensorflow.lite.examples.accelerationservice.validator.MeanSquaredErrorValidator;

/** Sample activity used for Acceleration Service tests. */
public class MainActivity extends AppCompatActivity {

  /** Maximum Mean-Squared-Error threshold used for accuracy validation. */
  public static final double MSE_THRESHOLD = 0.003;

  public final Context context = MainActivity.this;

  private final Executor executor = Executors.newSingleThreadExecutor();

  private Logger logger;
  private AssetModelFactory assetModelFactory;
  private AccuracyValidator validator;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    TextView logOutputView = findViewById(R.id.log_output);
    logOutputView.setMovementMethod(new ScrollingMovementMethod());
    logger = new TextViewLogger(this, logOutputView);

    assetModelFactory = new AssetModelFactory(context, executor, logger);
    validator = new MeanSquaredErrorValidator(logger, MSE_THRESHOLD);

    findViewById(R.id.cpu_validation_plain_addition_model_btn)
        .setOnClickListener(
            v -> {
              logger.clear();
              logTaskFailures(runPlainAdditionCpuValidation());
            });
    findViewById(R.id.gpu_validation_plain_addition_model_btn)
        .setOnClickListener(
            v -> {
              logger.clear();
              logTaskFailures(runPlainAdditionGpuValidation());
            });

    findViewById(R.id.cpu_validation_mobilenet_model_btn)
        .setOnClickListener(
            v -> {
              logger.clear();
              logTaskFailures(runMobileNetV1CpuValidation());
            });
    findViewById(R.id.gpu_validation_mobilenet_model_btn)
        .setOnClickListener(
            v -> {
              logger.clear();
              logTaskFailures(runMobileNetV1GpuValidation());
            });
  }

  /**
   * Uses Acceleration Service to check if the {@code model} passes accuracy checks defined by the
   * {@code validationConfig} when running a benchmark on a configuration specified by the {@code
   * accelerationConfig}. If the {@code accelerationConfig} is safe and valid, it can be applied on
   * the {@link #InterpreterApi.Options} and passed to the {@link InterpreterApi}.
   */
  public Task<Boolean> runScenario(
      Executor executor,
      AssetModel assetModel,
      AccelerationConfig accelerationConfig,
      ValidationConfig validationConfig) {

    // 1. Initialize TFLite in Google Play services.
    Task<Void> initializeTask = initializeTfLite();

    return initializeTask
        .onSuccessTask(
            aVoid -> {
              // 2. Validate the acceleration config.
              logger.info("TFLite initialized successfully.");
              Model model = assetModel.getModel();
              AccelerationService service = AccelerationService.create(context);

              return service.validateConfig(model, accelerationConfig, validationConfig);
            })
        .onSuccessTask(
            executor,
            validatedAccelerationConfig -> {
              logger.info("Validated acceleration config result: " + validatedAccelerationConfig);

              // 3. Check if validated config is safe and applies it on interpreter options.
              Options options = createInterpreterOptions(validatedAccelerationConfig);

              // 4. Check if the benchmark output is valid.
              boolean validBenchmarkOutput =
                  validateBenchmarkOutputs(
                      assetModel, validatedAccelerationConfig.benchmarkResult().actualOutput());
              logger.info(
                  "AccelerationService: Benchmark model output is valid: " + validBenchmarkOutput);

              // 5. Create an interpreter and run inference using sample input.
              boolean validInterpreterOutput = runInference(assetModel, options);

              // 6. Return true if benchmark output and interpreter outputs are valid.
              return Tasks.forResult(validBenchmarkOutput && validInterpreterOutput);
            })
        .addOnSuccessListener(aVoid -> logger.info("Scenario successful!"));
  }

  private Task<Boolean> runPlainAdditionCpuValidation() {
    logger.info("Running CPU validation test on Plain Addition model.");
    Task<AssetModel> model = assetModelFactory.load(ModelType.PLAIN_ADDITION);
    AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder().build();
    return runValidation(model, accelerationConfig);
  }

  private Task<Boolean> runMobileNetV1CpuValidation() {
    logger.info("Running CPU validation test on MobileNetV1 model.");
    Task<AssetModel> model = assetModelFactory.load(ModelType.MOBILENET_V1);
    AccelerationConfig accelerationConfig = new CpuAccelerationConfig.Builder().build();
    return runValidation(model, accelerationConfig);
  }

  private Task<Boolean> runPlainAdditionGpuValidation() {
    logger.info("Running GPU validation test on Plain Addition model.");
    Task<AssetModel> model = assetModelFactory.load(ModelType.PLAIN_ADDITION);
    AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder().build();
    return runValidation(model, accelerationConfig);
  }

  private Task<Boolean> runMobileNetV1GpuValidation() {
    logger.info("Running GPU validation test on MobileNetV1 model.");
    Task<AssetModel> model = assetModelFactory.load(ModelType.MOBILENET_V1);
    AccelerationConfig accelerationConfig = new GpuAccelerationConfig.Builder().build();
    return runValidation(model, accelerationConfig);
  }

  private Task<Boolean> runValidation(
      Task<AssetModel> modelTask, AccelerationConfig accelerationConfig) {
    return modelTask.onSuccessTask(
        model -> {
          ValidationConfig validationConfig =
              new CustomValidationConfig.Builder()
                  .setGoldenInputs(model.getInputs())
                  .setAccuracyValidator(validator)
                  .setBatchSize(model.getBatchSize())
                  .build();
          logger.info("Starting validation scenario.");
          return runScenario(executor, model, accelerationConfig, validationConfig);
        });
  }

  @WorkerThread
  private boolean runInference(AssetModel assetModel, InterpreterApi.Options options) {
    ByteBuffer model = assetModel.getModel().modelBuffer();
    Object[] inputs = assetModel.getInputs();
    Map<Integer, Object> outputs = assetModel.allocateOutputs();
    try (InterpreterApi interpreter = InterpreterApi.create(model, options)) {
      interpreter.runForMultipleInputsOutputs(inputs, outputs);
    }
    return assetModel.validateInterpreterOutputs(getValues(outputs));
  }

  private Task<Void> initializeTfLite() {
    return TfLiteGpu.isGpuDelegateAvailable(context)
        .onSuccessTask(
            gpuAvailable -> {
              findViewById(R.id.gpu_validation_plain_addition_model_btn).setEnabled(gpuAvailable);
              findViewById(R.id.gpu_validation_mobilenet_model_btn).setEnabled(gpuAvailable);
              return TfLite.initialize(
                  context,
                  TfLiteInitializationOptions.builder()
                      .setEnableGpuDelegateSupport(gpuAvailable)
                      .build());
            });
  }

  private boolean validateBenchmarkOutputs(
      AssetModel assetModel, List<InferenceOutput> benchmarkOutputs) {
    int size = benchmarkOutputs.size();
    ByteBuffer[] outputs = new ByteBuffer[size];
    for (int i = 0; i < size; i++) {
      outputs[i] = benchmarkOutputs.get(i).getValue();
    }
    return assetModel.validateBenchmarkOutputs(outputs);
  }

  private Options createInterpreterOptions(ValidatedAccelerationConfigResult result) {
    Options options = new Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
    if (!result.isValid()
        || result.benchmarkResult() == null
        || !result.benchmarkResult().hasPassedAccuracyCheck()) {
      // If the validation failed, do not apply acceleration config to the interpreter options.
      logger.info(
          "Acceleration config is not safe! Creating Interpreter Options without acceleration"
              + " config.");
      if (result.benchmarkError() != null) {
        logger.info("Benchmark error: " + result.benchmarkError());
      }
    } else {
      for (BenchmarkMetric benchmarkMetric : result.benchmarkResult().metrics()) {
        logger.info("Metric name: " + benchmarkMetric.getName());
        logger.info("Metric values: ");
        for (float val : benchmarkMetric.getValues()) {
          logger.info(String.valueOf(val));
        }
        options.setAccelerationConfig(result);
      }
    }
    return options;
  }

  private Object[] getValues(Map<Integer, Object> map) {
    return map.values().toArray(new Object[0]);
  }

  private void logTaskFailures(Task<Boolean> task) {
    task.addOnSuccessListener(isValid -> logger.info("Scenario passed: " + isValid))
        .addOnFailureListener(e -> logger.error("Scenario failed", e));
  }
}
