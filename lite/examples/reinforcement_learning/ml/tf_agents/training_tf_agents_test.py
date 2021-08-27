# Lint as: python3
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
"""End-to-end tests for Plane Strike TF training."""
import os
import tensorflow.compat.v1 as tf_v1
from tensorflow_examples.lite.examples.reinforcement_learning.ml.tf_agents import training_tf_agents

gfile = tf_v1.io.gfile


class TrainingTFAgentsTest(tf_v1.test.TestCase):

  def setUp(self):
    super(TrainingTFAgentsTest, self).setUp()
    self.tmp_dir = self.get_temp_dir()
    self.model_path = os.path.join(self.tmp_dir, 'planestrike_tf_agents.tflite')
    if os.path.exists(self.model_path):
      os.remove(self.model_path)

  def test_e2e(self):
    training_tf_agents.train_agent(1, self.tmp_dir,
                                   os.path.join(self.tmp_dir, 'tf_agents_log'),
                                   self.tmp_dir)
    self.assertTrue(gfile.exists(self.model_path))

  def tearDown(self):
    super(TrainingTFAgentsTest, self).tearDown()
    if os.path.exists(self.model_path):
      os.remove(self.model_path)


if __name__ == '__main__':
  tf_v1.test.main()
