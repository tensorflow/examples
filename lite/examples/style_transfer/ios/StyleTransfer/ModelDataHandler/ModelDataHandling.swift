//
//  ModelDataHandling.swift
//  StyleTransfer
//
//  Created by Ivan Cheung on 3/28/20.
//  Copyright © 2020 Ivan Cheung. All rights reserved.
//

import CoreImage

/// A result from invoking the `Interpreter`.
struct Result<Inference> {
  let elapsedTimeInMs: Double
  let inference: Inference
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

protocol ModelDataHandling {
    associatedtype Input
    associatedtype Inference
    
    var threadCount: Int { get }
    
    func runModel(input: Input) -> Result<Inference>?
}
