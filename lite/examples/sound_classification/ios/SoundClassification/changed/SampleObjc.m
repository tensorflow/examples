//
//  SampleObjc.m
//  SoundClassification
//
//  Created by Prianka Kariat on 01/05/22.
//  Copyright © 2022 Tensorflow. All rights reserved.
//

#import "SampleObjc.h"
#import "TFLAudioClassifier.h"

@implementation SampleObjc

-(void)newMeth {
  
 TFLAudioClassifier *classifier = [[TFLAudioClassifier alloc] initWithModelPath:@"hello" error:nil];
  
  TFLAudioClassifierOptions *oopt = [[TFLAudioClassifierOptions alloc] initWithModelPath:@"hello"];
  
 TFLAudioClassifier *classifierO = [[TFLAudioClassifier alloc] initWithOptions:oopt error:nil];
  NSLog(@"%@",classifierO);
  NSLog(@"%@", classifier);
//  [TFLAudioClassifier alloc] initWithModelPath:(nonnull NSString *) error:(NSError *__autoreleasing  _Nullable * _Nullable)
//
}

@end
