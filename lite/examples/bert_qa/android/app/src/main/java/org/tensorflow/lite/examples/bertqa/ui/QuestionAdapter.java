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
import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.support.design.chip.Chip;
import org.tensorflow.lite.examples.bertqa.R;

/** Adapter class to show question suggestion chips. */
public class QuestionAdapter extends RecyclerView.Adapter<QuestionAdapter.MyViewHolder> {

  private LayoutInflater inflater;
  private String[] questions;
  private OnQuestionSelectListener onQuestionSelectListener;

  public QuestionAdapter(Context context, String[] questions) {
    inflater = LayoutInflater.from(context);
    this.questions = questions;
  }

  @Override
  public QuestionAdapter.MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

    View view = inflater.inflate(R.layout.question_chip, parent, false);
    MyViewHolder holder = new MyViewHolder(view);

    return holder;
  }

  @Override
  public void onBindViewHolder(QuestionAdapter.MyViewHolder holder, int position) {
    holder.chip.setText(questions[position]);
    holder.chip.setOnClickListener(
        view -> onQuestionSelectListener.onQuestionSelect(questions[position]));
  }

  @Override
  public int getItemCount() {
    return questions.length;
  }

  public void setOnQuestionSelectListener(OnQuestionSelectListener onQuestionSelectListener) {
    this.onQuestionSelectListener = onQuestionSelectListener;
  }

  class MyViewHolder extends RecyclerView.ViewHolder {

    Chip chip;

    public MyViewHolder(View itemView) {
      super(itemView);
      chip = itemView.findViewById(R.id.chip);
    }
  }

  /** Interface for callback when a question is selected. */
  public interface OnQuestionSelectListener {
    void onQuestionSelect(String question);
  }
}
