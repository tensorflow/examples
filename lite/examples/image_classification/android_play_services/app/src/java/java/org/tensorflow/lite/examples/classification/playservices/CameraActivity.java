/*
 * Copyright 2022 The TensorFlow Authors
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

package org.tensorflow.lite.examples.classification.playservices;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysis.Analyzer;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.java.TfLite;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.tensorflow.lite.examples.classification.playservices.ImageClassificationHelper
        .Recognition;
import org.tensorflow.lite.examples.classification.playservices.databinding.ActivityCameraBinding;

/** Activity that displays the camera and performs object detection on the incoming frames. */
public final class CameraActivity extends AppCompatActivity {

  private static final String TAG = "CameraActivity";
  // Number of recognition results to show in the UI
  private static final int MAX_REPORT = 3;
  private static final String PERMISSION = Manifest.permission.CAMERA;

  private final ExecutorService executor = Executors.newSingleThreadExecutor();

  private ActivityResultLauncher<String> requestPermissionLauncher;
  private ActivityCameraBinding activityCameraBinding;
  private Bitmap bitmapBuffer;
  private boolean pauseAnalysis = false;
  private int imageRotationDegrees = 0;
  // Initialize TFLite once
  private Task<Void> initializeTask;
  // The classifier is create after initialization succeeded
  private ImageClassificationHelper classifier;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    activityCameraBinding = ActivityCameraBinding.inflate(this.getLayoutInflater());
    setContentView(activityCameraBinding.getRoot());

    AtomicBoolean isGpuInitialized = new AtomicBoolean(false);

    if (initializeTask == null) {
          // Initialize TFLite asynchronously
          initializeTask = TfLite.initialize(
                          this,
                          TfLiteInitializationOptions
                                  .builder()
                                  .setEnableGpuDelegateSupport(true)
                                  .build())
                  .continueWithTask(task -> {
                      if (task.isSuccessful()) {
                          isGpuInitialized.set(true);
                          return Tasks.forResult(null);
                      } else {
                          // Fallback to initialize interpreter without GPU
                          isGpuInitialized.set(false);
                          return TfLite.initialize(CameraActivity.this);
                      }
                  })
                  .addOnSuccessListener(unused -> {
                      startInitialization(isGpuInitialized.get());
                  })
                  .addOnFailureListener(err -> {
                      Log.e(TAG, "Failed to initialize the classifier.", err);
                  });
      }

    // Request for permission
    requestPermissionLauncher = requestPermission();

    // Set up camera
    activityCameraBinding.cameraCaptureButton.setOnClickListener(setUpCameraCaptureButton());

  }

  private void startInitialization(boolean isGpuInitialized) {
    Log.d(TAG, "TFLite in Play Services initialized successfully.");
    // Create ImageClassificationHelper AFTER TfLite.initialize() succeeded. This
    // guarantees that all the interactions with TFLite happens after initialization.
    try {
      classifier = ImageClassificationHelper.create(
              CameraActivity.this,
              MAX_REPORT,
              isGpuInitialized
      );
    } catch (Exception e) {
      Log.d(TAG, "ImageClassificationHelper initialization error");
    }
  }


  @Override
  protected void onDestroy() {
    // Terminate all outstanding analyzing jobs (if there is any)
    executor.shutdown();
    try {
      if (!executor.awaitTermination(1000, TimeUnit.MILLISECONDS)) {
        Log.w(TAG, "Failed to terminate.");
      }
    } catch (InterruptedException e) {
      Log.e(TAG, "Exit was interrupted.", e);
    }

    // Release TFLite resources
    if (classifier != null) {
      classifier.close();
    }
    super.onDestroy();
  }

  @Override
  protected void onResume() {
    super.onResume();

    // Request permissions each time the app resumes, since they can be revoked at any time
    if (!hasPermission(this)) {
      requestPermissionLauncher.launch(PERMISSION);
    } else {
      bindCameraUseCases();
    }
  }

  /** Declare and bind preview and analysis use cases */
  @SuppressLint("UnsafeExperimentalUsageError")
  private void bindCameraUseCases() {
    activityCameraBinding.viewFinder.post(
            () -> {
              ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                      ProcessCameraProvider.getInstance(this);
              cameraProviderFuture.addListener(
                      () -> {
                        // Camera provider is now guaranteed to be available
                        ProcessCameraProvider cameraProvider;
                        try {
                          cameraProvider = cameraProviderFuture.get();
                        } catch (ExecutionException | InterruptedException e) {
                          Log.e(TAG, "Failed to get Camera.", e);
                          return;
                        }

                        // Set up the view finder use case to display camera preview
                        Preview preview =
                                new Preview.Builder()
                                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                                        .setTargetRotation(
                                                activityCameraBinding
                                                        .viewFinder
                                                        .getDisplay()
                                                        .getRotation())
                                        .build();

                        // Set up the image analysis use case which will process frames in real time
                        ImageAnalysis imageAnalysis =
                                new ImageAnalysis.Builder()
                                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                                        .setTargetRotation(
                                                activityCameraBinding.viewFinder
                                                        .getDisplay()
                                                        .getRotation()
                                        )
                                        .setBackpressureStrategy(
                                                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                                        )
                                        .setOutputImageFormat(
                                                ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                                        )
                                        .build();

                        imageAnalysis.setAnalyzer(executor, new ClassificationAnalyzer());

                        // Apply declared configs to CameraX using the same lifecycle owner
                        cameraProvider.unbindAll();
                        cameraProvider.bindToLifecycle(
                                this,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalysis
                        );

                        // Use the camera object to link our preview use case with the view
                        preview.setSurfaceProvider(
                                activityCameraBinding
                                        .viewFinder
                                        .getSurfaceProvider()
                        );
                      },
                      ContextCompat.getMainExecutor(this));
            });
  }

  /** Image Analyzer used for classifying image. */
  private class ClassificationAnalyzer implements Analyzer {
    private int frameCounter = 0;
    private long lastFpsTimeMillis = System.currentTimeMillis();

    @Override
    public void analyze(@NonNull ImageProxy image) {
      if (bitmapBuffer == null) {
        // The image rotation and RGB image buffer are initialized only after the analyzer has
        // started running
        imageRotationDegrees = image.getImageInfo().getRotationDegrees();
        bitmapBuffer =
                Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
      }

      // Early exit: image analysis is in paused state, or TFLite initialization has not finished
      if (pauseAnalysis || classifier == null) {
        image.close();
        return;
      }

      // Copy out RGB bits to our shared buffer
      bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());

      // Perform the image classification for the current frame
      List<Recognition> recognitions = classifier.classify(bitmapBuffer, imageRotationDegrees);

      reportRecognition(recognitions);

      // Compute the FPS of the entire pipeline
      int frameCount = 10;
      if (++frameCounter % frameCount == 0) {
        frameCounter = 0;
        long currentTimeMillis = System.currentTimeMillis();
        long deltaTimeMillis = currentTimeMillis - lastFpsTimeMillis;
        float fps = 1000 * (float) frameCount / deltaTimeMillis;
        Log.d(TAG, "FPS: " + fps);
        lastFpsTimeMillis = currentTimeMillis;
      }
      image.close();
    }
  }

  /** Displays recognition results on screen. */
  private void reportRecognition(List<Recognition> recognitions) {
    activityCameraBinding.viewFinder.post(
            () -> {
              // Early exit: if recognition is empty
              if (recognitions.isEmpty()) {
                activityCameraBinding.textPrediction.setVisibility(View.GONE);
                return;
              }

              // Update the text and UI
              StringBuilder text = new StringBuilder();
              for (Recognition recognition : recognitions) {
                text.append(
                        String.format(
                                Locale.getDefault(),
                                "%.2f %s\n",
                                recognition.getConfidence(),
                                recognition.getTitle()));
              }
              activityCameraBinding.textPrediction.setText(text);

              // Make sure all UI elements are visible
              activityCameraBinding.textPrediction.setVisibility(View.VISIBLE);
            });
  }

  /** Returns the callback used when clicking the camera capture button. */
  private View.OnClickListener setUpCameraCaptureButton() {
    return listener -> {
      // Disable all camera controls
      listener.setEnabled(false);
      if (pauseAnalysis) {
        // If image analysis is in paused state, resume it
        pauseAnalysis = false;
        activityCameraBinding.imagePredicted.setVisibility(View.GONE);
      } else {
        // Otherwise, pause image analysis and freeze image
        pauseAnalysis = true;
        Matrix matrix = new Matrix();
        matrix.postRotate(imageRotationDegrees);
        Bitmap uprightImage =
                Bitmap.createBitmap(
                        bitmapBuffer,
                        0,
                        0,
                        bitmapBuffer.getWidth(),
                        bitmapBuffer.getHeight(),
                        matrix,
                        true);
        activityCameraBinding.imagePredicted.setImageBitmap(uprightImage);
        activityCameraBinding.imagePredicted.setVisibility(View.VISIBLE);
      }

      // Re-enable camera controls
      listener.setEnabled(true);
    };
  }

  /** Registers request permission callback. */
  private ActivityResultLauncher<String> requestPermission() {
    return registerForActivityResult(
            new RequestPermission(),
            isGranted -> {
              if (isGranted) {
                bindCameraUseCases();
              } else {
                finish(); // If we don't have the required permissions, we can't run
              }
            });
  }

  /** Convenience method used to check if all permissions required by this app are granted. */
  private boolean hasPermission(Context context) {
    return ContextCompat.checkSelfPermission(context, PERMISSION)
            == PackageManager.PERMISSION_GRANTED;
  }
}
