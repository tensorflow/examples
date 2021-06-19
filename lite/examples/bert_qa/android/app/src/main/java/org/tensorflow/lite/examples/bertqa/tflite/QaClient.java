package org.tensorflow.lite.examples.bertqa.tflite;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.task.text.qa.BertQuestionAnswerer;
import org.tensorflow.lite.task.text.qa.QaAnswer;


import java.io.IOException;
import java.util.List;

/** Load TfLite model and provide predictions with task api. */
public class QaClient {

    private static final String TAG = "TaskApi";
    private static final String MODEL_PATH = "model.tflite";

    private final Context context;

    public BertQuestionAnswerer answerer;

    public QaClient(Context context) {
        this.context = context;
    }

    /** Load TF Lite model. */
    public void loadModel() {
        try {
            answerer = BertQuestionAnswerer.createFromFile(context, MODEL_PATH);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }
    }

    /** Free up resources as the client is no longer needed. */
    public void unload() {
        answerer.close();
        answerer = null;
    }


    /** Run inference and predict the possible answers. */
    public List<QaAnswer> predict(String questionToAsk, String contextOfTheQuestion) {

        List<QaAnswer> apiResult = answerer.answer(contextOfTheQuestion, questionToAsk);
        return apiResult;
    }
}
