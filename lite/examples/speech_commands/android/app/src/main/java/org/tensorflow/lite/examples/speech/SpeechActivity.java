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

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import androidx.appcompat.widget.SwitchCompat;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.google.android.material.bottomsheet.BottomSheetBehavior;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.tensorflow.lite.Interpreter;

/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity
    implements View.OnClickListener, CompoundButton.OnCheckedChangeListener {

  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.
  private static final int SAMPLE_RATE = 16000;
  private static final int SAMPLE_DURATION_MS = 1000;
  private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
  private static final long AVERAGE_WINDOW_DURATION_MS = 1000;
  private static final float DETECTION_THRESHOLD = 0.50f;
  private static final int SUPPRESSION_MS = 1500;
  private static final int MINIMUM_COUNT = 3;
  private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
  private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
  private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.tflite";

  private static final String HANDLE_THREAD_NAME = "CameraBackground";

  // UI elements.
  private static final int REQUEST_RECORD_AUDIO = 13;
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  // Working variables.
  short[] recordingBuffer = new short[RECORDING_LENGTH];
  int recordingOffset = 0;
  boolean shouldContinue = true;
  private Thread recordingThread;
  boolean shouldContinueRecognition = true;
  private Thread recognitionThread;
  private final ReentrantLock recordingBufferLock = new ReentrantLock();
  private final ReentrantLock tfLiteLock = new ReentrantLock();

  private List<String> labels = new ArrayList<String>();
  private List<String> displayedLabels = new ArrayList<>();
  private RecognizeCommands recognizeCommands = null;
  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private final Interpreter.Options tfLiteOptions = new Interpreter.Options();
  private MappedByteBuffer tfLiteModel;
  private Interpreter tfLite;
  private ImageView bottomSheetArrowImageView;

  private TextView yesTextView,
      noTextView,
      upTextView,
      downTextView,
      leftTextView,
      rightTextView,
      onTextView,
      offTextView,
      stopTextView,
      goTextView;
  private TextView sampleRateTextView, inferenceTimeTextView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_sc_activity_speech);

    // Load the labels for the model, but only display those that don't start
    // with an underscore.
    String actualLabelFilename = LABEL_FILENAME.split("file:///android_asset/", -1)[1];
    Log.i(LOG_TAG, "Reading labels from: " + actualLabelFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(getAssets().open(actualLabelFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
        if (line.charAt(0) != '_') {
          displayedLabels.add(line.substring(0, 1).toUpperCase() + line.substring(1));
        }
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }

    // Set up an object to smooth recognition results to increase accuracy.
    recognizeCommands =
        new RecognizeCommands(
            labels,
            AVERAGE_WINDOW_DURATION_MS,
            DETECTION_THRESHOLD,
            SUPPRESSION_MS,
            MINIMUM_COUNT,
            MINIMUM_TIME_BETWEEN_SAMPLES_MS);

    String actualModelFilename = MODEL_FILENAME.split("file:///android_asset/", -1)[1];
    try {
      tfLiteModel = loadModelFile(getAssets(), actualModelFilename);
      recreateInterpreter();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Start the recording and recognition threads.
    requestMicrophonePermission();
    startRecording();
    startRecognition();

    sampleRateTextView = findViewById(R.id.sample_rate);
    inferenceTimeTextView = findViewById(R.id.inference_info);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);

    yesTextView = findViewById(R.id.yes);
    noTextView = findViewById(R.id.no);
    upTextView = findViewById(R.id.up);
    downTextView = findViewById(R.id.down);
    leftTextView = findViewById(R.id.left);
    rightTextView = findViewById(R.id.right);
    onTextView = findViewById(R.id.on);
    offTextView = findViewById(R.id.off);
    stopTextView = findViewById(R.id.stop);
    goTextView = findViewById(R.id.go);

    apiSwitchCompat.setOnCheckedChangeListener(this);

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            int height = gestureLayout.getMeasuredHeight();

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
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
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
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    sampleRateTextView.setText(SAMPLE_RATE + " Hz");
  }

  private void requestMicrophonePermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      requestPermissions(
          new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_RECORD_AUDIO
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      startRecording();
      startRecognition();
    }
  }

  public synchronized void startRecording() {
    if (recordingThread != null) {
      return;
    }
    shouldContinue = true;
    recordingThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                record();
              }
            });
    recordingThread.start();
  }

  public synchronized void stopRecording() {
    if (recordingThread == null) {
      return;
    }
    shouldContinue = false;
    recordingThread = null;
  }

  private void record() {
    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

    // Estimate the buffer size we'll need for this device.
    int bufferSize =
        AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
      bufferSize = SAMPLE_RATE * 2;
    }
    short[] audioBuffer = new short[bufferSize / 2];

    AudioRecord record =
        new AudioRecord(
            MediaRecorder.AudioSource.DEFAULT,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize);

    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }

    record.startRecording();

    Log.v(LOG_TAG, "Start recording");

    // Loop, gathering audio data and copying it to a round-robin buffer.
    while (shouldContinue) {
      int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
      int maxLength = recordingBuffer.length;
      int newRecordingOffset = recordingOffset + numberRead;
      int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
      int firstCopyLength = numberRead - secondCopyLength;
      // We store off all the data for the recognition thread to access. The ML
      // thread will copy out of this buffer into its own, while holding the
      // lock, so this should be thread safe.
      recordingBufferLock.lock();
      try {
        System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
        System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
        recordingOffset = newRecordingOffset % maxLength;
      } finally {
        recordingBufferLock.unlock();
      }
    }

    record.stop();
    record.release();
  }

  public synchronized void startRecognition() {
    if (recognitionThread != null) {
      return;
    }
    shouldContinueRecognition = true;
    recognitionThread =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                recognize();
              }
            });
    recognitionThread.start();
  }

  public synchronized void stopRecognition() {
    if (recognitionThread == null) {
      return;
    }
    shouldContinueRecognition = false;
    recognitionThread = null;
  }

  private void recognize() {

    Log.v(LOG_TAG, "Start recognition");

    short[] inputBuffer = new short[RECORDING_LENGTH];
    float[][] floatInputBuffer = new float[RECORDING_LENGTH][1];
    float[][] outputScores = new float[1][labels.size()];
    int[] sampleRateList = new int[] {SAMPLE_RATE};

    // Loop, grabbing recorded data and running the recognition model on it.
    while (shouldContinueRecognition) {
      long startTime = new Date().getTime();
      // The recording thread places data in this round-robin buffer, so lock to
      // make sure there's no writing happening and then copy it to our own
      // local version.
      recordingBufferLock.lock();
      try {
        int maxLength = recordingBuffer.length;
        int firstCopyLength = maxLength - recordingOffset;
        int secondCopyLength = recordingOffset;
        System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
        System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
      } finally {
        recordingBufferLock.unlock();
      }

      // We need to feed in float values between -1.0f and 1.0f, so divide the
      // signed 16-bit inputs.
      for (int i = 0; i < RECORDING_LENGTH; ++i) {
        floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
      }

      Object[] inputArray = {floatInputBuffer, sampleRateList};
      Map<Integer, Object> outputMap = new HashMap<>();
      outputMap.put(0, outputScores);

      // Run the model.
      tfLiteLock.lock();
      try {
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
      } finally {
        tfLiteLock.unlock();
      }

      // Use the smoother to figure out if we've had a real recognition event.
      long currentTime = System.currentTimeMillis();
      final RecognizeCommands.RecognitionResult result =
          recognizeCommands.processLatestResults(outputScores[0], currentTime);
      lastProcessingTimeMs = new Date().getTime() - startTime;
      runOnUiThread(
          new Runnable() {
            @Override
            public void run() {

              inferenceTimeTextView.setText(lastProcessingTimeMs + " ms");

              // If we do have a new command, highlight the right list entry.
              if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                int labelIndex = -1;
                for (int i = 0; i < labels.size(); ++i) {
                  if (labels.get(i).equals(result.foundCommand)) {
                    labelIndex = i;
                  }
                }

                switch (labelIndex - 2) {
                  case 0:
                    selectedTextView = yesTextView;
                    break;
                  case 1:
                    selectedTextView = noTextView;
                    break;
                  case 2:
                    selectedTextView = upTextView;
                    break;
                  case 3:
                    selectedTextView = downTextView;
                    break;
                  case 4:
                    selectedTextView = leftTextView;
                    break;
                  case 5:
                    selectedTextView = rightTextView;
                    break;
                  case 6:
                    selectedTextView = onTextView;
                    break;
                  case 7:
                    selectedTextView = offTextView;
                    break;
                  case 8:
                    selectedTextView = stopTextView;
                    break;
                  case 9:
                    selectedTextView = goTextView;
                    break;
                }

                if (selectedTextView != null) {
                  selectedTextView.setBackgroundResource(R.drawable.round_corner_text_bg_selected);
                  final String score = Math.round(result.score * 100) + "%";
                  selectedTextView.setText(selectedTextView.getText() + "\n" + score);
                  selectedTextView.setTextColor(
                      getResources().getColor(android.R.color.holo_orange_light));
                  handler.postDelayed(
                      new Runnable() {
                        @Override
                        public void run() {
                          String origionalString =
                              selectedTextView.getText().toString().replace(score, "").trim();
                          selectedTextView.setText(origionalString);
                          selectedTextView.setBackgroundResource(
                              R.drawable.round_corner_text_bg_unselected);
                          selectedTextView.setTextColor(
                              getResources().getColor(android.R.color.darker_gray));
                        }
                      },
                      750);
                }
              }
            }
          });
      try {
        // We don't need to run too frequently, so snooze for a bit.
        Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);
      } catch (InterruptedException e) {
        // Ignore
      }
    }

    Log.v(LOG_TAG, "End recognition");
  }

  @Override
  public void onClick(View v) {
    if ((v.getId() != R.id.plus) && (v.getId() != R.id.minus)) {
      return;
    }

    String threads = threadsTextView.getText().toString().trim();
    int numThreads = Integer.parseInt(threads);
    if (v.getId() == R.id.plus) {
      numThreads++;
    } else {
      if (numThreads == 1) {
        return;
      }
      numThreads--;
    }

    final int finalNumThreads = numThreads;
    threadsTextView.setText(String.valueOf(finalNumThreads));
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setNumThreads(finalNumThreads);
          recreateInterpreter();
        });
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setUseNNAPI(isChecked);
          recreateInterpreter();
        });
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  private void recreateInterpreter() {
    tfLiteLock.lock();
    try {
      if (tfLite != null) {
        tfLite.close();
        tfLite = null;
      }
      tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
      tfLite.resizeInput(0, new int[] {RECORDING_LENGTH, 1});
      tfLite.resizeInput(1, new int[] {1});
    } finally {
      tfLiteLock.unlock();
    }
  }

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();

    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }
}
