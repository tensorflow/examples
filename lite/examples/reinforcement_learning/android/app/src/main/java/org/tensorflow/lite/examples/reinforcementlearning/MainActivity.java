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

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.GridView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

/** The main activity to provide interactions with users. */
public class MainActivity extends AppCompatActivity {

  private int agentHits;
  private int playerHits;
  private final BoardCellStatus[][] playerBoard =
      new BoardCellStatus[Constants.BOARD_SIZE][Constants.BOARD_SIZE];
  private final HiddenBoardCellStatus[][] playerHiddenBoard =
      new HiddenBoardCellStatus[Constants.BOARD_SIZE][Constants.BOARD_SIZE];
  private final BoardCellStatus[][] agentBoard =
      new BoardCellStatus[Constants.BOARD_SIZE][Constants.BOARD_SIZE];
  private final HiddenBoardCellStatus[][] agentHiddenBoard =
      new HiddenBoardCellStatus[Constants.BOARD_SIZE][Constants.BOARD_SIZE];

  private GridView agentBoardGridView;
  private GridView playerBoardGridView;
  private TextView agentHitsTextView;
  private TextView playerHitsTextView;
  private Button resetButton;

  private PolicyGradientAgent policyGradientAgent;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    agentBoardGridView = (GridView) findViewById(R.id.agent_board_gridview);
    playerBoardGridView = (GridView) findViewById(R.id.player_board_gridview);
    agentHitsTextView = (TextView) findViewById(R.id.agent_hits_textview);
    playerHitsTextView = (TextView) findViewById(R.id.player_hits_textview);
    initGame();
    try {
      policyGradientAgent = new PolicyGradientAgent(this);
    } catch (IOException e) {
      Log.e(Constants.TAG, e.getMessage());
      return;
    }

    playerBoardGridView.setAdapter(
        new BoardCellAdapter(this, playerBoard, playerHiddenBoard, false));
    agentBoardGridView.setAdapter(new BoardCellAdapter(this, agentBoard, agentHiddenBoard, true));
    agentBoardGridView.setOnItemClickListener(
        new AdapterView.OnItemClickListener() {
          @Override
          public void onItemClick(AdapterView<?> adapterView, View view, int position, long l) {
            // Player action
            int playerActionX = position / Constants.BOARD_SIZE;
            int playerActionY = position % Constants.BOARD_SIZE;
            if (agentBoard[playerActionX][playerActionY] == BoardCellStatus.UNTRIED) {
              if (agentHiddenBoard[playerActionX][playerActionY]
                  == HiddenBoardCellStatus.OCCUPIED_BY_PLANE) {
                agentBoard[playerActionX][playerActionY] = BoardCellStatus.HIT;
                playerHits++;
                playerHitsTextView.setText("Agent board:\n" + playerHits + " hits");
              } else {
                agentBoard[playerActionX][playerActionY] = BoardCellStatus.MISS;
              }
            }

            // Agent action
            StrikePrediction agentStrikePosition = policyGradientAgent.predictNextMove(playerBoard);
            if (!agentStrikePosition.isPredictedByAgent) {
              Toast.makeText(
                      MainActivity.this,
                      "Something went wrong with the RL agent! Continuing with a random strike",
                      Toast.LENGTH_LONG)
                  .show();
            }
            int agentStrikePositionX = agentStrikePosition.x;
            int agentStrikePositionY = agentStrikePosition.y;

            if (playerHiddenBoard[agentStrikePositionX][agentStrikePositionY]
                == HiddenBoardCellStatus.OCCUPIED_BY_PLANE) {
              // Hit
              playerBoard[agentStrikePositionX][agentStrikePositionY] = BoardCellStatus.HIT;
              agentHits++;
              agentHitsTextView.setText("Player board:\n" + agentHits + " hits");
            } else {
              // Miss
              playerBoard[agentStrikePositionX][agentStrikePositionY] = BoardCellStatus.MISS;
            }

            if (agentHits == Constants.PLANE_CELL_COUNT
                || playerHits == Constants.PLANE_CELL_COUNT) {
              // Game ends
              String gameEndMessage;
              if (agentHits == Constants.PLANE_CELL_COUNT
                  && playerHits == Constants.PLANE_CELL_COUNT) {
                gameEndMessage = "Draw game!";
              } else if (agentHits == Constants.PLANE_CELL_COUNT) {
                gameEndMessage = "Agent wins!";
              } else {
                gameEndMessage = "You win!";
              }
              Toast.makeText(MainActivity.this, gameEndMessage, Toast.LENGTH_LONG).show();
              // Automatically reset game UI after 2 seconds
              Timer resetGameTimer = new Timer();
              resetGameTimer.schedule(
                  new TimerTask() {
                    @Override
                    public void run() {
                      runOnUiThread(() -> initGame());
                    }
                  },
                  2000);
            }

            agentBoardGridView.invalidateViews();
            playerBoardGridView.invalidateViews();
          }
        });

    resetButton = (Button) findViewById(R.id.reset_button);
    resetButton.setOnClickListener(
        new View.OnClickListener() {
          @Override
          public void onClick(View view) {
            initGame();
          }
        });
  }

  private void initGame() {
    initiBoard(playerBoard);
    placePlaneOnHiddenBoard(playerHiddenBoard);
    initiBoard(agentBoard);
    placePlaneOnHiddenBoard(agentHiddenBoard);
    agentBoardGridView.invalidateViews();
    playerBoardGridView.invalidateViews();
    agentHits = 0;
    playerHits = 0;
    agentHitsTextView.setText("Player board:\n0 hits");
    playerHitsTextView.setText("Agent board:\n0 hits");
  }

  private void initiBoard(BoardCellStatus[][] board) {
    for (int i = 0; i < Constants.BOARD_SIZE; i++) {
      Arrays.fill(board[i], 0, Constants.BOARD_SIZE, BoardCellStatus.UNTRIED);
    }
  }

  private void initiHiddenBoard(HiddenBoardCellStatus[][] board) {
    for (int i = 0; i < Constants.BOARD_SIZE; i++) {
      Arrays.fill(board[i], 0, Constants.BOARD_SIZE, HiddenBoardCellStatus.UNOCCUPIED);
    }
  }

  private void placePlaneOnHiddenBoard(HiddenBoardCellStatus[][] hiddenBoard) {
    initiHiddenBoard(hiddenBoard);

    // Place the plane on the board
    // First, decide the plane's orientation
    //   0: heading right
    //   1: heading up
    //   2: heading left
    //   3: heading down
    Random rand = new Random();
    int planeOrientation = rand.nextInt(4);

    // Next, figure out the location of plane core as the '*' below
    //   | |      |      | |    ---
    //   |-*-    -*-    -*-|     |
    //   | |      |      | |    -*-
    //           ---             |
    int planeCoreX;
    int planeCoreY;
    switch (planeOrientation) {
      case 0:
        planeCoreX = rand.nextInt(Constants.BOARD_SIZE - 2) + 1;
        planeCoreY = rand.nextInt(Constants.BOARD_SIZE - 3) + 2;
        // Populate the tail
        hiddenBoard[planeCoreX][planeCoreY - 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX - 1][planeCoreY - 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX + 1][planeCoreY - 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        break;
      case 1:
        planeCoreX = rand.nextInt(Constants.BOARD_SIZE - 3) + 1;
        planeCoreY = rand.nextInt(Constants.BOARD_SIZE - 2) + 1;
        // Populate the tail
        hiddenBoard[planeCoreX + 2][planeCoreY] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX + 2][planeCoreY + 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX + 2][planeCoreY - 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        break;
      case 2:
        planeCoreX = rand.nextInt(Constants.BOARD_SIZE - 2) + 1;
        planeCoreY = rand.nextInt(Constants.BOARD_SIZE - 3) + 1;
        // Populate the tail
        hiddenBoard[planeCoreX][planeCoreY + 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX - 1][planeCoreY + 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX + 1][planeCoreY + 2] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        break;
      default:
        planeCoreX = rand.nextInt(Constants.BOARD_SIZE - 3) + 2;
        planeCoreY = rand.nextInt(Constants.BOARD_SIZE - 2) + 1;
        // Populate the tail
        hiddenBoard[planeCoreX - 2][planeCoreY] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX - 2][planeCoreY + 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
        hiddenBoard[planeCoreX - 2][planeCoreY - 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
    }

    // Finally, populate the 'cross' in the plane
    hiddenBoard[planeCoreX][planeCoreY] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
    hiddenBoard[planeCoreX + 1][planeCoreY] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
    hiddenBoard[planeCoreX - 1][planeCoreY] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
    hiddenBoard[planeCoreX][planeCoreY + 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
    hiddenBoard[planeCoreX][planeCoreY - 1] = HiddenBoardCellStatus.OCCUPIED_BY_PLANE;
  }
}
