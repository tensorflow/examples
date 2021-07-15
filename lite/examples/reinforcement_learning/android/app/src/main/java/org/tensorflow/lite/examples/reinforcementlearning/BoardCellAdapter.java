/*
 * Copyright 2020 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.reinforcementlearning;

import android.content.Context;
import android.graphics.Color;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

/** The gridview adapter for filling the board. */
public class BoardCellAdapter extends BaseAdapter {
  // We always use square board, so only one size is needed
  private static final int BOARD_SIZE = 8;

  private final Context context;
  private final BoardCellStatus[][] board;
  private final HiddenBoardCellStatus[][] hiddenBoard;
  private final boolean isAgentBoard;

  public BoardCellAdapter(
      Context context,
      BoardCellStatus[][] board,
      HiddenBoardCellStatus[][] hiddenBoard,
      boolean isAgentBoard) {
    this.context = context;
    this.board = board;
    this.hiddenBoard = hiddenBoard;
    this.isAgentBoard = isAgentBoard;
  }

  @Override
  public View getView(int position, View convertView, ViewGroup parent) {
    TextView cellTextView = new TextView(context);

    int x = position / BOARD_SIZE;
    int y = position % BOARD_SIZE;

    if (board[x][y] == BoardCellStatus.UNTRIED) {
      // Untried cell
      cellTextView.setBackgroundColor(Color.WHITE);
      if (hiddenBoard[x][y] == HiddenBoardCellStatus.OCCUPIED_BY_PLANE && !isAgentBoard) {
        cellTextView.setBackgroundColor(Color.BLUE);
      }
    } else if (board[x][y] == BoardCellStatus.HIT) {
      // Hit
      cellTextView.setBackgroundColor(Color.RED);
    } else {
      // Miss
      cellTextView.setBackgroundColor(Color.YELLOW);
    }

    cellTextView.setHeight(80);

    return cellTextView;
  }

  @Override
  public int getCount() {
    return BOARD_SIZE * BOARD_SIZE;
  }

  @Override
  public Object getItem(int position) {
    return null;
  }

  @Override
  public long getItemId(int position) {
    return 0;
  }
}
