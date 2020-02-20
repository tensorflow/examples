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
package org.tensorflow.lite.examples.bertqa.ui;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.text.Editable;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.TextWatcher;
import android.text.method.ScrollingMovementMethod;
import android.text.style.BackgroundColorSpan;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.ImageButton;
import android.widget.TextView;
import android.support.design.widget.Snackbar;
import android.support.design.widget.TextInputEditText;
import java.util.List;
import java.util.Locale;
import org.tensorflow.lite.examples.bertqa.R;
import org.tensorflow.lite.examples.bertqa.ml.LoadDatasetClient;
import org.tensorflow.lite.examples.bertqa.ml.QaAnswer;
import org.tensorflow.lite.examples.bertqa.ml.QaClient;

/** Activity for doing Q&A on a specific dataset */
public class QaActivity extends AppCompatActivity {

  private static final String DATASET_POSITION_KEY = "DATASET_POSITION";
  private static final String TAG = "QaActivity";
  private static final boolean DISPLAY_RUNNING_TIME = false;

  private TextInputEditText questionEditText;
  private TextView contentTextView;
  private TextToSpeech textToSpeech;

  private boolean questionAnswered = false;
  private String content;
  private Handler handler;
  private QaClient qaClient;

  public static Intent newInstance(Context context, int datasetPosition) {
    Intent intent = new Intent(context, QaActivity.class);
    intent.putExtra(DATASET_POSITION_KEY, datasetPosition);
    return intent;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    Log.v(TAG, "onCreate");
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_qa_activity_qa);

    // Get content of the selected dataset.
    int datasetPosition = getIntent().getIntExtra(DATASET_POSITION_KEY, -1);
    LoadDatasetClient datasetClient = new LoadDatasetClient(this);

    // Show the dataset title.
    TextView titleText = findViewById(R.id.title_text);
    titleText.setText(datasetClient.getTitles()[datasetPosition]);

    // Show the text content of the selected dataset.
    content = datasetClient.getContent(datasetPosition);
    contentTextView = findViewById(R.id.content_text);
    contentTextView.setText(content);
    contentTextView.setMovementMethod(new ScrollingMovementMethod());

    // Setup question suggestion list.
    RecyclerView questionSuggestionsView = findViewById(R.id.suggestion_list);
    QuestionAdapter adapter =
        new QuestionAdapter(this, datasetClient.getQuestions(datasetPosition));
    adapter.setOnQuestionSelectListener(question -> answerQuestion(question));
    questionSuggestionsView.setAdapter(adapter);
    LinearLayoutManager layoutManager =
        new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false);
    questionSuggestionsView.setLayoutManager(layoutManager);

    // Setup ask button.
    ImageButton askButton = findViewById(R.id.ask_button);
    askButton.setOnClickListener(
        view -> answerQuestion(questionEditText.getText().toString()));

    // Setup text edit where users can input their question.
    questionEditText = findViewById(R.id.question_edit_text);
    questionEditText.setOnFocusChangeListener(
        (view, hasFocus) -> {
          // If we already answer current question, clear the question so that user can input a new
          // one.
          if (hasFocus && questionAnswered) {
            questionEditText.setText(null);
          }
        });
    questionEditText.addTextChangedListener(
        new TextWatcher() {
          @Override
          public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}

          @Override
          public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
            // Only allow clicking Ask button if there is a question.
            boolean shouldAskButtonActive = !charSequence.toString().isEmpty();
            askButton.setClickable(shouldAskButtonActive);
            askButton.setImageResource(
                shouldAskButtonActive ? R.drawable.ic_ask_active : R.drawable.ic_ask_inactive);
          }

          @Override
          public void afterTextChanged(Editable editable) {}
        });
    questionEditText.setOnKeyListener(
        (v, keyCode, event) -> {
          if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_ENTER) {
            answerQuestion(questionEditText.getText().toString());
          }
          return false;
        });

    // Setup QA client to and background thread to run inference.
    HandlerThread handlerThread = new HandlerThread("QAClient");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
    qaClient = new QaClient(this);
  }

  @Override
  protected void onStart() {
    Log.v(TAG, "onStart");
    super.onStart();
    handler.post(
        () -> {
          qaClient.loadModel();
          qaClient.loadDictionary();
        });

    textToSpeech =
        new TextToSpeech(
            this,
            status -> {
              if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
              } else {
                textToSpeech = null;
              }
            });
  }

  @Override
  protected void onStop() {
    Log.v(TAG, "onStop");
    super.onStop();
    handler.post(() -> qaClient.unload());

    if (textToSpeech != null) {
      textToSpeech.stop();
      textToSpeech.shutdown();
    }
  }

  private void answerQuestion(String question) {
    question = question.trim();
    if (question.isEmpty()) {
      questionEditText.setText(question);
      return;
    }

    // Append question mark '?' if not ended with '?'.
    // This aligns with question format that trains the model.
    if (!question.endsWith("?")) {
      question += '?';
    }
    final String questionToAsk = question;
    questionEditText.setText(questionToAsk);

    // Delete all pending tasks.
    handler.removeCallbacksAndMessages(null);

    // Hide keyboard and dismiss focus on text edit.
    InputMethodManager imm =
        (InputMethodManager) getSystemService(AppCompatActivity.INPUT_METHOD_SERVICE);
    imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
    View focusView = getCurrentFocus();
    if (focusView != null) {
      focusView.clearFocus();
    }

    // Reset content text view
    contentTextView.setText(content);

    questionAnswered = false;

    Snackbar runningSnackbar =
        Snackbar.make(contentTextView, "Looking up answer...", Integer.MAX_VALUE);
    runningSnackbar.show();

    // Run TF Lite model to get the answer.
    handler.post(
        () -> {
          long beforeTime = System.currentTimeMillis();
          final List<QaAnswer> answers = qaClient.predict(questionToAsk, content);
          long afterTime = System.currentTimeMillis();
          double totalSeconds = (afterTime - beforeTime) / 1000.0;

          if (!answers.isEmpty()) {
            // Get the top answer
            QaAnswer topAnswer = answers.get(0);
            // Show the answer.
            runOnUiThread(
                () -> {
                  runningSnackbar.dismiss();
                  presentAnswer(topAnswer);

                  String displayMessage = "Top answer was successfully highlighted.";
                  if (DISPLAY_RUNNING_TIME) {
                    displayMessage = String.format("%s %.3fs.", displayMessage, totalSeconds);
                  }
                  Snackbar.make(contentTextView, displayMessage, Snackbar.LENGTH_LONG).show();
                  questionAnswered = true;
                });
          }
        });
  }

  private void presentAnswer(QaAnswer answer) {
    // Highlight answer.
    Spannable spanText = new SpannableString(content);
    int offset = content.indexOf(answer.text, 0);
    if (offset >= 0) {
      spanText.setSpan(
          new BackgroundColorSpan(getColor(R.color.tfe_qa_color_highlight)),
          offset,
          offset + answer.text.length(),
          Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
    }
    contentTextView.setText(spanText);

    // Use TTS to speak out the answer.
    if (textToSpeech != null) {
      textToSpeech.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text);
    }
  }
}
