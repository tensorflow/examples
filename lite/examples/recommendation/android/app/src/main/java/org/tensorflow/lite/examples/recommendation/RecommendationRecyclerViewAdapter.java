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
package org.tensorflow.lite.examples.recommendation;

import androidx.recyclerview.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import java.util.List;
import org.tensorflow.lite.examples.recommendation.RecommendationClient.Result;
import org.tensorflow.lite.examples.recommendation.RecommendationFragment.OnListFragmentInteractionListener;
import org.tensorflow.lite.examples.recommendation.data.MovieItem;

/**
 * RecommendationRecyclerViewAdapter: a {@link RecyclerView.Adapter} that can display a recommended
 * {@link MovieItem} and makes a call to the specified {@link OnListFragmentInteractionListener}.
 */
public class RecommendationRecyclerViewAdapter
    extends RecyclerView.Adapter<RecommendationRecyclerViewAdapter.ViewHolder> {

  private final List<Result> results;
  private final OnListFragmentInteractionListener listener;

  public RecommendationRecyclerViewAdapter(
      List<Result> results, OnListFragmentInteractionListener listener) {
    this.results = results;
    this.listener = listener;
  }

  @Override
  public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
    View view =
        LayoutInflater.from(parent.getContext())
            .inflate(R.layout.tfe_re_fragment_recommendation, parent, false);
    return new ViewHolder(view);
  }

  @Override
  public void onBindViewHolder(final ViewHolder holder, int position) {
    MovieItem item = results.get(position).item;
    holder.result = results.get(position);
    holder.recommendationMovieTitleView.setText(item.title + " - " + item.genres);
    holder.scoreView.setText(String.format("[%d]", item.id));

    holder.view.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View v) {
            if (null != listener) {
              // Notify the active callbacks interface (the activity, if the
              // fragment is attached to one) that an item has been selected.
              // listener.onListFragmentInteraction(holder.mItem);
              listener.onClickRecommendedMovie(item);
            }
          }
        });
  }

  @Override
  public int getItemCount() {
    return results.size();
  }

  /** ViewHolder to display one movie in list view of recommendation result. */
  public static class ViewHolder extends RecyclerView.ViewHolder {

    public final View view;
    public final TextView scoreView;
    public final TextView recommendationMovieTitleView;
    public Result result;

    public ViewHolder(View view) {
      super(view);
      this.view = view;
      this.scoreView = (TextView) view.findViewById(R.id.recommendation_score);
      this.recommendationMovieTitleView =
          (TextView) view.findViewById(R.id.recommendation_movie_title);
    }

    @Override
    public String toString() {
      return super.toString() + " '" + recommendationMovieTitleView.getText() + "'";
    }
  }
}
