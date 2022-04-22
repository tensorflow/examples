//
//  ViewController.swift
//  TFLAudioRecord
//
//  Created by Prianka Kariat on 22/04/22.
//

import UIKit


class ViewController: UIViewController {
  let audioRecord = TFLAudioRecord(channelCount: 2, sampleRate: 8000, bufferSize: 800);

  override func viewDidLoad() {
    super.viewDidLoad()
    
    audioRecord.checkAndStartTappingMicrophone { buffer, error in
       print(buffer!.size)
      if error != nil {
        print("Error");
      }
      for i in 0..<buffer!.size {
        if(buffer!.data[i] != 0.0) {
          print("Yes")
        }
        print(buffer!.data[i])
      }
    }
    
//    let inputManager = AudioInputManager(sampleRate: 8000);
//    inputManager.checkPermissionsAndStartTappingMicrophone()

    // Do any additional setup after loading the view.
//    print("Hello 1");
//    let status = try! audioRecord.checkAndStartTappingMicrophone()
//    GMLAudio *audio = GMLAudio(
    
//    audioRecord.checkPermissionAndStartTappingMicrophoneWithError(NSErrorPointer);
    
//    audioRecord.checkPermissionAndStartTappingMicrophone { error in
//
//      if error != nil {
//        print(error!);
//      }
//      print("Worked");
//    }
    

  }


}

