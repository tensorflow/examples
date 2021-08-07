/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.icu.text.SimpleDateFormat;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.databinding.DataBindingUtil;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.examples.detection.databinding.TfeOdActivityCameraBinding;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

public class CameraActivity extends AppCompatActivity implements View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;

  private boolean debug = false;

  private Handler handler;
  private HandlerThread handlerThread;
  private boolean firstTimeStartModel = true;
  private boolean isProcessingFrame = false;

  private LinearLayout bottomSheetLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private enum DetectorMode {
    TF_OD_API;
  }

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final Size DESIRED_ANALYSIS_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private long lastProcessingTimeMs;

  protected ImageView bottomSheetArrowImageView;

  private Integer sensorOrientation;

  private Detector detector;

  //Data Binding
  private TfeOdActivityCameraBinding binding;

  private long timestamp = 0;
  private MultiBoxTracker tracker;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    binding = DataBindingUtil.setContentView(this, R.layout.tfe_od_activity_camera);

    if (hasPermission()) {
      //Start CameraX
      startCamera();
    } else {
      //Requesting Permission for CameraX
      requestPermission();
    }

    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    //Controlling bottom modal sheet
    ViewTreeObserver vto = binding.bottomSheetLayout.gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                binding.bottomSheetLayout.gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int height = binding.bottomSheetLayout.gestureLayout.getMeasuredHeight();
                sheetBehavior.setPeekHeight(height);
              }
            });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_DRAGGING:
                    break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }

              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {
              }
            });

    binding.bottomSheetLayout.plus.setOnClickListener(this);
    binding.bottomSheetLayout.minus.setOnClickListener(this);
  }

  private void onStartCameraX(Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    BorderedText borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);
    Log.v("Camera Image Rotation", String.valueOf(rotation));

    sensorOrientation = rotation;
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    tracker = new MultiBoxTracker(this);

    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      this,
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      setUseNNAPI(true);
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);
    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);

    binding.trackingOverlay.addCallback(
            canvas -> {
              tracker.draw(canvas);
              if (isDebug()) {
                tracker.drawDebug(canvas);
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @SuppressLint("UnsafeOptInUsageError")
  private void startCamera() {
    ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

    cameraProviderFuture.addListener(() -> {
      try {
        ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();

        //Selecting the Camera here - Back Camera
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        //Images are processed by passing an executor in which the image analysis is run
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), image -> {
          int rotationDegrees = image.getImageInfo().getRotationDegrees();
          Log.i("Rotation Degrees", String.valueOf(rotationDegrees));
          Log.i("Rotation preview", String.valueOf(binding.previewView.getDisplay().getRotation()));

          ++timestamp;
          final long currTimestamp = timestamp;

          if (firstTimeStartModel) {
            onStartCameraX(DESIRED_ANALYSIS_SIZE, rotationDegrees);
            firstTimeStartModel = false;
          }

          if (!isProcessingFrame) {
            runInBackground(
                    () -> {
                      if (detector != null) {
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Detector.Recognition> results = detector.recognizeImage(image.getImage(), sensorOrientation);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.e("Degrees: %s", results);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        if (MODE == DetectorMode.TF_OD_API) {
                          minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        }

                        final List<Detector.Recognition> mappedRecognitions =
                                new ArrayList<>();

                        for (final Detector.Recognition result : results) {
                          final RectF location = result.getLocation();
                          if (location != null && result.getConfidence() >= minimumConfidence) {
                            result.setLocation(location);
                            mappedRecognitions.add(result);
                          }
                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        binding.trackingOverlay.postInvalidate();

                        runOnUiThread(
                                () -> {
                                  showFrameInfo(DESIRED_ANALYSIS_SIZE.getWidth() + "x" + DESIRED_ANALYSIS_SIZE.getHeight());
                                  showCropInfo(TF_OD_API_INPUT_SIZE + "x" + TF_OD_API_INPUT_SIZE);
                                  showInference(lastProcessingTimeMs + "ms");
                                });
                      }
                      image.close();
                      isProcessingFrame = false;
                    });
            isProcessingFrame = true;
          }
        });

        // Connect the preview use case to the previewView
        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

        // Attach use cases to the camera with the same lifecycle owner
        if (cameraProvider != null) {
          Camera camera = cameraProvider.bindToLifecycle(
                  this,
                  cameraSelector,
                  imageAnalysis,
                  preview);
        }

      } catch (ExecutionException | InterruptedException e) {
        e.printStackTrace();
      }
    }, ContextCompat.getMainExecutor(this));
  }

  public boolean isDebug() {
    return debug;
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }


  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        //Start CameraX
        startCamera();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
                .show();
      }
      requestPermissions(new String[]{PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  protected void showFrameInfo(String frameInfo) {
    binding.bottomSheetLayout.frameInfo.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    binding.bottomSheetLayout.cropInfo.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    binding.bottomSheetLayout.inferenceInfo.setText(inferenceTime);
  }

  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
            () -> {
              try {
                detector.setUseNNAPI(isChecked);
              } catch (UnsupportedOperationException e) {
                LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                runOnUiThread(
                        () -> {
                          Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                        });
              }
            });
  }

  private void setNumThreads(final int numThreads) {
    runInBackground(() -> {
      detector.setNumThreads(numThreads);
    });
  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      numThreads++;
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    } else if (v.getId() == R.id.minus) {
      String threads = binding.bottomSheetLayout.threads.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) {
        return;
      }
      numThreads--;
      binding.bottomSheetLayout.threads.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }
  }
}