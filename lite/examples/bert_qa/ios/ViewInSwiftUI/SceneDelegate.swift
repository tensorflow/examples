// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftUI
import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

  var window: UIWindow?

  func scene(
    _ scene: UIScene, willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    // Use a UIHostingController as window root view controller
    if let windowScene = scene as? UIWindowScene {
      let bertQA: BertQAHandler
      do {
        bertQA = try BertQAHandler()
      } catch let error {
        fatalError(error.localizedDescription)
      }

      let window = UIWindow(windowScene: windowScene)
      window.rootViewController = UIHostingController(
        rootView: DatasetListView(datasets: Dataset.load(), bertQA: bertQA))
      self.window = window
      window.makeKeyAndVisible()
    }
  }
}
