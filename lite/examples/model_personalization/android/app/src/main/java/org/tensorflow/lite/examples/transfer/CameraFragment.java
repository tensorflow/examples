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

package org.tensorflow.lite.examples.transfer;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.camera.core.CameraX;
import androidx.camera.core.CameraX.LensFacing;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysis.ImageReaderMode;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.ImageProxy.PlaneProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.databinding.BindingAdapter;
import androidx.databinding.DataBindingUtil;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProviders;
import com.google.android.material.chip.Chip;
import com.google.android.material.chip.ChipGroup;
import com.google.android.material.snackbar.Snackbar;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;
import org.tensorflow.lite.examples.transfer.databinding.CameraFragmentBinding;

/**
 * The main fragment of the classifier.
 *
 * Camera functionality (through CameraX) is heavily based on the official example:
 * https://github.com/android/camera/tree/master/CameraXBasic.
 */
public class CameraFragment extends Fragment {

  private static final int LOWER_BYTE_MASK = 0xFF;

  private static final String TAG = CameraFragment.class.getSimpleName();

  private static final LensFacing LENS_FACING = LensFacing.BACK;

  private TextureView viewFinder;

  private Integer viewFinderRotation = null;

  private Size bufferDimens = new Size(0, 0);
  private Size viewFinderDimens = new Size(0, 0);

  private CameraFragmentViewModel viewModel;
  private TransferLearningModelWrapper tlModel;

  // When the user presses the "add sample" button for some class,
  // that class will be added to this queue. It is later extracted by
  // InferenceThread and processed.
  private final ConcurrentLinkedQueue<String> addSampleRequests = new ConcurrentLinkedQueue<>();

  private final LoggingBenchmark inferenceBenchmark = new LoggingBenchmark("InferenceBench");

  /**
   * Set up a responsive preview for the view finder.
   */
  private void startCamera() {
    viewFinderRotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
    if (viewFinderRotation == null) {
      viewFinderRotation = 0;
    }

    DisplayMetrics metrics = new DisplayMetrics();
    viewFinder.getDisplay().getRealMetrics(metrics);
    Rational screenAspectRatio = new Rational(metrics.widthPixels, metrics.heightPixels);

    PreviewConfig config = new PreviewConfig.Builder()
        .setLensFacing(LENS_FACING)
        .setTargetAspectRatio(screenAspectRatio)
        .setTargetRotation(viewFinder.getDisplay().getRotation())
        .build();

    Preview preview = new Preview(config);

    preview.setOnPreviewOutputUpdateListener(previewOutput -> {
      ViewGroup parent = (ViewGroup) viewFinder.getParent();
      parent.removeView(viewFinder);
      parent.addView(viewFinder, 0);

      viewFinder.setSurfaceTexture(previewOutput.getSurfaceTexture());

      Integer rotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
      updateTransform(rotation, previewOutput.getTextureSize(), viewFinderDimens);
    });

    viewFinder.addOnLayoutChangeListener((
        view, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom) -> {
      Size newViewFinderDimens = new Size(right - left, bottom - top);
      Integer rotation = getDisplaySurfaceRotation(viewFinder.getDisplay());
      updateTransform(rotation, bufferDimens, newViewFinderDimens);
    });

    HandlerThread inferenceThread = new HandlerThread("InferenceThread");
    inferenceThread.start();
    ImageAnalysisConfig analysisConfig = new ImageAnalysisConfig.Builder()
        .setLensFacing(LENS_FACING)
        .setCallbackHandler(new Handler(inferenceThread.getLooper()))
        .setImageReaderMode(ImageReaderMode.ACQUIRE_LATEST_IMAGE)
        .setTargetRotation(viewFinder.getDisplay().getRotation())
        .build();

    ImageAnalysis imageAnalysis = new ImageAnalysis(analysisConfig);
    imageAnalysis.setAnalyzer(inferenceAnalyzer);

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  private final ImageAnalysis.Analyzer inferenceAnalyzer =
      (imageProxy, rotationDegrees) -> {
        final String imageId = UUID.randomUUID().toString();

        inferenceBenchmark.startStage(imageId, "preprocess");
        float[] rgbImage = prepareCameraImage(yuvCameraImageToBitmap(imageProxy), rotationDegrees);
        inferenceBenchmark.endStage(imageId, "preprocess");

        // Adding samples is also handled by inference thread / use case.
        // We don't use CameraX ImageCapture since it has very high latency (~650ms on Pixel 2 XL)
        // even when using .MIN_LATENCY.
        String sampleClass = addSampleRequests.poll();
        if (sampleClass != null) {
          inferenceBenchmark.startStage(imageId, "addSample");
          try {
            tlModel.addSample(rgbImage, sampleClass).get();
          } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
          } catch (InterruptedException e) {
            // no-op
          }

          viewModel.increaseNumSamples(sampleClass);
          inferenceBenchmark.endStage(imageId, "addSample");

        } else {
          // We don't perform inference when adding samples, since we should be in capture mode
          // at the time, so the inference results are not actually displayed.
          inferenceBenchmark.startStage(imageId, "predict");
          Prediction[] predictions = tlModel.predict(rgbImage);
          if (predictions == null) {
            return;
          }
          inferenceBenchmark.endStage(imageId, "predict");

          for (Prediction prediction : predictions) {
            viewModel.setConfidence(prediction.getClassName(), prediction.getConfidence());
          }
        }

        inferenceBenchmark.finish(imageId);
      };

  public final View.OnClickListener onAddSampleClickListener = view -> {
    String className;
    if (view.getId() == R.id.class_btn_1) {
      className = "1";
    } else if (view.getId() == R.id.class_btn_2) {
      className = "2";
    } else if (view.getId() == R.id.class_btn_3) {
      className = "3";
    } else if (view.getId() == R.id.class_btn_4) {
      className = "4";
    } else {
      throw new RuntimeException("Listener called for unexpected view");
    }

    addSampleRequests.add(className);
  };

  /**
   * Fit the camera preview into [viewFinder].
   *
   * @param rotation view finder rotation.
   * @param newBufferDimens camera preview dimensions.
   * @param newViewFinderDimens view finder dimensions.
   */
  private void updateTransform(Integer rotation, Size newBufferDimens, Size newViewFinderDimens) {
    if (Objects.equals(rotation, viewFinderRotation)
      && Objects.equals(newBufferDimens, bufferDimens)
      && Objects.equals(newViewFinderDimens, viewFinderDimens)) {
      return;
    }

    if (rotation == null) {
      return;
    } else {
      viewFinderRotation = rotation;
    }

    if (newBufferDimens.getWidth() == 0 || newBufferDimens.getHeight() == 0) {
      return;
    } else {
      bufferDimens = newBufferDimens;
    }

    if (newViewFinderDimens.getWidth() == 0 || newViewFinderDimens.getHeight() == 0) {
      return;
    } else {
      viewFinderDimens = newViewFinderDimens;
    }

    Log.d(TAG, String.format("Applying output transformation.\n"
        + "View finder size: %s.\n"
        + "Preview output size: %s\n"
        + "View finder rotation: %s\n", viewFinderDimens, bufferDimens, viewFinderRotation));
    Matrix matrix = new Matrix();

    float centerX = viewFinderDimens.getWidth() / 2f;
    float centerY = viewFinderDimens.getHeight() / 2f;

    matrix.postRotate(-viewFinderRotation.floatValue(), centerX, centerY);

    float bufferRatio = bufferDimens.getHeight() / (float) bufferDimens.getWidth();

    int scaledWidth;
    int scaledHeight;
    if (viewFinderDimens.getWidth() > viewFinderDimens.getHeight()) {
      scaledHeight = viewFinderDimens.getWidth();
      scaledWidth = Math.round(viewFinderDimens.getWidth() * bufferRatio);
    } else {
      scaledHeight = viewFinderDimens.getHeight();
      scaledWidth = Math.round(viewFinderDimens.getHeight() * bufferRatio);
    }

    float xScale = scaledWidth / (float) viewFinderDimens.getWidth();
    float yScale = scaledHeight / (float) viewFinderDimens.getHeight();

    matrix.preScale(xScale, yScale, centerX, centerY);

    viewFinder.setTransform(matrix);
  }

  @Override
  public void onCreate(Bundle bundle) {
    super.onCreate(bundle);

    tlModel = new TransferLearningModelWrapper(getActivity());
    viewModel = ViewModelProviders.of(this).get(CameraFragmentViewModel.class);
    viewModel.setTrainBatchSize(tlModel.getTrainBatchSize());
  }

  @Override
  public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle bundle) {
    CameraFragmentBinding dataBinding =
        DataBindingUtil.inflate(inflater, R.layout.camera_fragment, container, false);
    dataBinding.setLifecycleOwner(getViewLifecycleOwner());
    dataBinding.setVm(viewModel);
    View rootView = dataBinding.getRoot();

    for (int buttonId : new int[] {
        R.id.class_btn_1, R.id.class_btn_2, R.id.class_btn_3, R.id.class_btn_4}) {
      rootView.findViewById(buttonId).setOnClickListener(onAddSampleClickListener);
    }

    ChipGroup chipGroup = (ChipGroup) rootView.findViewById(R.id.mode_chip_group);
    if (viewModel.getCaptureMode().getValue()) {
      ((Chip) rootView.findViewById(R.id.capture_mode_chip)).setChecked(true);
    } else {
      ((Chip) rootView.findViewById(R.id.inference_mode_chip)).setChecked(true);
    }

    chipGroup.setOnCheckedChangeListener((group, checkedId) -> {
      if (checkedId == R.id.capture_mode_chip) {
        viewModel.setCaptureMode(true);
      } else if (checkedId == R.id.inference_mode_chip) {
        viewModel.setCaptureMode(false);
      }
    });

    return dataBinding.getRoot();
  }

  @Override
  public void onViewCreated(View view, Bundle bundle) {
    super.onViewCreated(view, bundle);

    viewFinder = getActivity().findViewById(R.id.view_finder);
    viewFinder.post(this::startCamera);
  }

  @Override
  public void onActivityCreated(Bundle bundle) {
    super.onActivityCreated(bundle);

    viewModel
        .getTrainingState()
        .observe(
            getViewLifecycleOwner(),
            trainingState -> {
              switch (trainingState) {
                case STARTED:
                  tlModel.enableTraining((epoch, loss) -> viewModel.setLastLoss(loss));
                  if (!viewModel.getInferenceSnackbarWasDisplayed().getValue()) {
                    Snackbar.make(
                            getActivity().findViewById(R.id.classes_bar),
                            R.string.switch_to_inference_hint,
                            Snackbar.LENGTH_LONG)
                        .show();
                    viewModel.markInferenceSnackbarWasCalled();
                  }
                  break;
                case PAUSED:
                  tlModel.disableTraining();
                  break;
                case NOT_STARTED:
                  break;
              }
            });
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    tlModel.close();
    tlModel = null;
  }

  private static Integer getDisplaySurfaceRotation(Display display) {
    if (display == null) {
      return null;
    }

    switch (display.getRotation()) {
      case Surface.ROTATION_0: return 0;
      case Surface.ROTATION_90: return 90;
      case Surface.ROTATION_180: return 180;
      case Surface.ROTATION_270: return 270;
      default: return null;
    }
  }

  private static Bitmap yuvCameraImageToBitmap(ImageProxy imageProxy) {
    if (imageProxy.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          "Expected a YUV420 image, but got " + imageProxy.getFormat());
    }

    PlaneProxy yPlane = imageProxy.getPlanes()[0];
    PlaneProxy uPlane = imageProxy.getPlanes()[1];

    int width = imageProxy.getWidth();
    int height = imageProxy.getHeight();

    byte[][] yuvBytes = new byte[3][];
    int[] argbArray = new int[width * height];
    for (int i = 0; i < imageProxy.getPlanes().length; i++) {
      final ByteBuffer buffer = imageProxy.getPlanes()[i].getBuffer();
      yuvBytes[i] = new byte[buffer.capacity()];
      buffer.get(yuvBytes[i]);
    }

    ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0],
        yuvBytes[1],
        yuvBytes[2],
        width,
        height,
        yPlane.getRowStride(),
        uPlane.getRowStride(),
        uPlane.getPixelStride(),
        argbArray);

    return Bitmap.createBitmap(argbArray, width, height, Config.ARGB_8888);
  }

  /**
   * Normalizes a camera image to [0; 1], cropping it
   * to size expected by the model and adjusting for camera rotation.
   */
  private static float[] prepareCameraImage(Bitmap bitmap, int rotationDegrees)  {
    int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;

    Bitmap paddedBitmap = padToSquare(bitmap);
    Bitmap scaledBitmap = Bitmap.createScaledBitmap(
        paddedBitmap, modelImageSize, modelImageSize, true);

    Matrix rotationMatrix = new Matrix();
    rotationMatrix.postRotate(rotationDegrees);
    Bitmap rotatedBitmap = Bitmap.createBitmap(
        scaledBitmap, 0, 0, modelImageSize, modelImageSize, rotationMatrix, false);

    float[] normalizedRgb = new float[modelImageSize * modelImageSize * 3];
    int nextIdx = 0;
    for (int y = 0; y < modelImageSize; y++) {
      for (int x = 0; x < modelImageSize; x++) {
        int rgb = rotatedBitmap.getPixel(x, y);

        float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.f);
        float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.f);
        float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.f);

        normalizedRgb[nextIdx++] = r;
        normalizedRgb[nextIdx++] = g;
        normalizedRgb[nextIdx++] = b;
      }
    }

    return normalizedRgb;
  }

  private static Bitmap padToSquare(Bitmap source) {
    int width = source.getWidth();
    int height = source.getHeight();

    int paddingX = width < height ? (height - width) / 2 : 0;
    int paddingY = height < width ? (width - height) / 2 : 0;
    Bitmap paddedBitmap = Bitmap.createBitmap(
        width + 2 * paddingX, height + 2 * paddingY, Config.ARGB_8888);
    Canvas canvas = new Canvas(paddedBitmap);
    canvas.drawARGB(0xFF, 0xFF, 0xFF, 0xFF);
    canvas.drawBitmap(source, paddingX, paddingY, null);
    return paddedBitmap;
  }

  // Binding adapters:

  @BindingAdapter({"captureMode", "inferenceText", "captureText"})
  public static void setClassSubtitleText(
      TextView view, boolean captureMode, Float inferenceText, Integer captureText) {
    if (captureMode) {
      view.setText(captureText != null ? Integer.toString(captureText) : "0");
    } else {
      view.setText(
          String.format(Locale.getDefault(), "%.2f", inferenceText != null ? inferenceText : 0.f));
    }
  }

  @BindingAdapter({"android:visibility"})
  public static void setViewVisibility(View view, boolean visible) {
    view.setVisibility(visible ? View.VISIBLE : View.GONE);
  }

  @BindingAdapter({"highlight"})
  public static void setClassButtonHighlight(View view, boolean highlight) {
    int drawableId;
    if (highlight) {
      drawableId = R.drawable.btn_default_highlight;
    } else {
      drawableId = R.drawable.btn_default;
    }
    view.setBackground(view.getContext().getDrawable(drawableId));
  }
}
