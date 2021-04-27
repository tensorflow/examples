#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for example_generation_movielens."""

import pandas as pd
import tensorflow as tf
from data import example_generation_movielens as example_gen

from google.protobuf import text_format


MOVIES_DF = pd.DataFrame([
    {
        'MovieId': int(1),
        'Title': 'Toy Story (1995)',
        'Genres': 'Animation|Children|Comedy'
    },
    {
        'MovieId': int(2),
        'Title': 'Four Weddings and a Funeral (1994)',
        'Genres': 'Comedy|Romance'
    },
    {
        'MovieId': int(3),
        'Title': 'Lion King, The (1994)',
        'Genres': 'Adventure|Animation|Children'
    },
    {
        'MovieId': int(4),
        'Title': 'Forrest Gump (1994)',
        'Genres': 'Comedy|Drama|Romance'
    },
    {
        'MovieId': int(5),
        'Title': 'Little Women (1994)',
        'Genres': 'Drama'
    },
])

RATINGS_DF = pd.DataFrame([
    {
        'UserID': int(1),
        'MovieId': int(1),
        'Rating': 3.5,
        'Timestamp': 0
    },
    {
        'UserID': int(1),
        'MovieId': int(2),
        'Rating': 4.0,
        'Timestamp': 1
    },
    {
        'UserID': int(1),
        'MovieId': int(3),
        'Rating': 4.5,
        'Timestamp': 2
    },
    {
        'UserID': int(1),
        'MovieId': int(4),
        'Rating': 4.7,
        'Timestamp': 3
    },
    {
        'UserID': int(2),
        'MovieId': int(5),
        'Rating': 4.0,
        'Timestamp': 1
    },
])

EXAMPLE1 = text_format.Parse(
    """
    features {
        feature {
          key: "context_movie_id"
          value {
            int64_list {
              value: [1, 0, 0, 0, 0]
            }
          }
        }
        feature {
          key: "context_movie_rating"
          value {
            float_list {
              value: [3.5, 0.0, 0.0, 0.0, 0.0]
            }
          }
        }
        feature {
          key: "context_movie_genre"
          value {
            bytes_list {
              value: ["Animation", "Children", "Comedy", "UNK", "UNK", "UNK", "UNK", "UNK", "UNK", "UNK"]
            }
          }
        }
        feature {
          key: "context_movie_year"
          value {
            int64_list {
              value: [1995, 0, 0, 0, 0]
            }
          }
        }
        feature {
          key: "label_movie_id"
          value {
            int64_list {
              value: [2]
            }
          }
        }
      }
      """, tf.train.Example())

EXAMPLE2 = text_format.Parse(
    """
    features {
        feature {
          key: "context_movie_id"
          value {
            int64_list {
              value: [1, 2, 0, 0, 0]
            }
          }
        }
        feature {
          key: "context_movie_rating"
          value {
            float_list {
              value: [3.5, 4.0, 0.0, 0.0, 0.0]
            }
          }
        }
        feature {
          key: "context_movie_genre"
          value {
            bytes_list {
              value: [
                    "Animation", "Children", "Comedy", "Comedy", "Romance", "UNK", "UNK", "UNK", "UNK", "UNK"
                ]
            }
          }
        }
        feature {
          key: "context_movie_year"
          value {
            int64_list {
              value: [1995, 1994, 0, 0, 0]
            }
          }
        }
        feature {
          key: "label_movie_id"
          value {
            int64_list {
              value: [3]
            }
          }
        }
      }""", tf.train.Example())

EXAMPLE3 = text_format.Parse(
    """
    features {
        feature {
          key: "context_movie_id"
          value {
            int64_list {
              value: [1, 2, 3, 0, 0]
            }
          }
        }
        feature {
          key: "context_movie_rating"
          value {
            float_list {
              value: [3.5, 4.0, 4.5, 0.0, 0.0]
            }
          }
        }
        feature {
          key: "context_movie_genre"
          value {
            bytes_list {
              value: [
                    "Animation", "Children", "Comedy", "Comedy", "Romance",
                    "Adventure", "Animation", "Children", "UNK", "UNK"
                ]
            }
          }
        }
        feature {
          key: "context_movie_year"
          value {
            int64_list {
              value: [1995, 1994, 1994, 0, 0]
            }
          }
        }
        feature {
          key: "label_movie_id"
          value {
            int64_list {
              value: [4]
            }
          }
        }
      }""", tf.train.Example())


class ExampleGenerationMovielensTest(tf.test.TestCase):

  def test_example_generation(self):
    timelines, _ = example_gen.convert_to_timelines(RATINGS_DF)
    train_examples, test_examples = example_gen.generate_examples_from_timelines(
        timelines=timelines,
        movies_df=MOVIES_DF,
        min_timeline_len=2,
        max_context_len=5,
        max_context_movie_genre_len=10,
        train_data_fraction=0.66,
        shuffle=False)
    self.assertLen(train_examples, 2)
    self.assertLen(test_examples, 1)
    self.assertProtoEquals(train_examples[0], EXAMPLE1)
    self.assertProtoEquals(train_examples[1], EXAMPLE2)
    self.assertProtoEquals(test_examples[0], EXAMPLE3)

  def test_vocabs_generation(self):
    _, movie_counts = example_gen.convert_to_timelines(RATINGS_DF)
    (_, movie_year_vocab, movie_genre_vocab) = (
        example_gen.generate_movie_feature_vocabs(
            movies_df=MOVIES_DF, movie_counts=movie_counts))
    self.assertAllEqual(movie_year_vocab, [0, 1994, 1995])
    self.assertAllEqual(movie_genre_vocab, [
        'UNK', 'Comedy', 'Animation', 'Children', 'Romance', 'Drama',
        'Adventure'
    ])


if __name__ == '__main__':
  tf.test.main()
