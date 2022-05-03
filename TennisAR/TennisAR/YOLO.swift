//
//  YOLO.swift
//  TennisAR
//
//  Created by Nicholas Novelle on 26.04.2022.
//

import Foundation
import UIKit
import CoreML

class YOLO {
    // params
    public static var inputWidth = 640
    public static let inputHeight = 640
    public static let maxBoundingBoxes = 10
    
    private static let SCORE_THRESHOLD = 0.2
    private static let NMS_THRESHOLD = 0.4
    private let CONFIDENCE_THRESHOLD: Float = 0.25
    
    // prediction type
    struct Prediction {
      let classIndex: Int
      let score: Float
      let rect: CGRect
    }
    
    let anchors: [[[Float]]] = [
        [[12.07812, 10.70312], [17.35938, 16.76562], [16.32812, 31.57812]],
        [[27.20312, 20.84375], [30.59375, 32.90625], [50.37500, 26.43750]],
        [[47.00000, 43.78125], [37.96875, 69.75000], [75.37500, 67.93750]],
    ]
    let strides: [Float] = [8, 16, 32]
    
    let model = yolov5s()
    
    @inline(__always) func offset(_ bxNum: Int, _ bx: Int, _ c: Int) -> Int {
      return bxNum*c + bx
    }
    
    
    /*
     let featureNames = output.featureNames
     var out: MLMultiArray
     for feature in featureNames{
         out = output.featureValue(for: feature)!.multiArrayValue!
         if out.shape.count == 3{
             let bxNum = Int(out.shape[1].int32Value)
             let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(out.dataPointer))
             for bx in 0..<bxNum {
                 //let tc = featurePointer[bxNum*4+bxNum]
                 let tc = out[0]
                 //out[[0, bx, 4] as [NSNumber]].floatValue
                 if tc < Float(CONFIDENCE_THRESHOLD){
                     //print(tc)
                     /*
                     let tx = Float(featurePointer[offset(bxNum, bx, 0)])
                     let ty = Float(featurePointer[offset(bxNum, bx, 1)])
                     let tw = Float(featurePointer[offset(bxNum, bx, 2)])
                     let th = Float(featurePointer[offset(bxNum, bx, 3)])
                     let tb = Float(featurePointer[offset(bxNum, bx, 5)])
                      */
                 }
             }
         }
     }
     */
    
    
    public func predict(image: CVPixelBuffer) -> Double! {
      /*if let output = try? model.prediction(input: image) {
          let timeStamp = CACurrentMediaTime()
          let a = computeBoundingBoxes(layers: [output.var_854, output.var_888, output.var_922, output.var_944])
          let elapsed = CACurrentMediaTime() - timeStamp
          return elapsed
      } else {
        return 1
      }*/
       return 0
    }
       
    public func computeBoundingBoxes(layers: [MLMultiArray]) -> [Prediction] {
        var predictions = [Prediction]()
        for (layer, features) in layers.enumerated() {
            if features.count < 5{
                convertFeaturestoPredictions(features: features, predictions: &predictions, layer: layer)
            }
        }
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: CONFIDENCE_THRESHOLD)
    }
    
    
    
    private func convertFeaturestoPredictions(features: MLMultiArray, predictions: inout [Prediction], layer: Int) {
      assert(features.shape[0].intValue == 1)  // Batch size of 1
      let boxesPerCell = features.shape[1].intValue
      let gridHeight = features.shape[2].intValue
      let gridWidth = features.shape[3].intValue
      let numClasses = features.shape[4].intValue - 5

      let boxStride = features.strides[1].intValue
      let yStride = features.strides[2].intValue
      let xStride = features.strides[3].intValue
      assert(features.strides[4].intValue == 1)  // The below code assumes a channel stride of 1.
      let gridSize = strides[layer]

      assert(features.dataType == MLMultiArrayDataType.float32) // Ensure 32 bit before using unsafe pointer.
      let featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(features.dataPointer))

      for b in 0..<boxesPerCell {
        let anchorW = anchors[layer][b][0]
        let anchorH = anchors[layer][b][1]
        for cy in 0..<gridHeight {
          for cx in 0..<gridWidth {
            let d = b*boxStride + cx*xStride + cy*yStride
            let tc = Float(featurePointer[d + 4])
            let confidence = sigmoid(tc)

            var classes = [Float](repeating: 0, count: numClasses)
            for c in 0..<numClasses {
              classes[c] = Float(featurePointer[d + 5 + c])
            }
            classes = softmax(classes)

            let (detectedClass, bestClassScore) = classes.argmax()
            let confidenceInClass = bestClassScore * confidence
            if confidenceInClass > CONFIDENCE_THRESHOLD {
              let tx = Float(featurePointer[d])
              let ty = Float(featurePointer[d + 1])
              let tw = Float(featurePointer[d + 2])
              let th = Float(featurePointer[d + 3])

              // Code converted from:
              // https://github.com/ultralytics/yolov5/blob/ae4261c7749ff644f45c66b79ecb1fff06437052/models/yolo.py
              // Inside Detect.forward
              let x = (sigmoid(tx) * 2 - 0.5 + Float(cx)) * gridSize
              let y = (sigmoid(ty) * 2 - 0.5 + Float(cy)) * gridSize
              let w = pow(sigmoid(tw) * 2.0, 2) * anchorW
              let h = pow(sigmoid(th) * 2.0, 2) * anchorH

              let rect = CGRect(
                  x: CGFloat(x - w/2),
                  y: CGFloat(y - h/2),
                  width: CGFloat(w),
                  height: CGFloat(h)
              )

              let prediction = Prediction(
                  classIndex: detectedClass,
                  score: confidenceInClass,
                  rect: rect
              )
              predictions.append(prediction)
            }
          }
        }
      }
    }
/*
    public func computeBoundingBoxes(features: MLMultiArray) -> [Prediction] {
      assert(features.count == 125*13*13)

      var predictions = [Prediction]()

      let blockSize: Float = 32
      let gridHeight = 13
      let gridWidth = 13
      let boxesPerCell = 5
      let numClasses = 20

      // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
      // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
      // five data items: x, y, width, height, and a confidence score. Each grid
      // cell also predicts which class each bounding box belongs to.
      //
      // The "features" array therefore contains (numClasses + 5)*boxesPerCell
      // values for each grid cell, i.e. 125 channels. The total features array
      // contains 125x13x13 elements.
      // NOTE: It turns out that accessing the elements in the multi-array as
      // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
      // It's much faster to use direct memory access to the features.
      let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
      let channelStride = features.strides[0].intValue
      let yStride = features.strides[1].intValue
      let xStride = features.strides[2].intValue

      @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
        return channel*channelStride + y*yStride + x*xStride
      }

      for cy in 0..<gridHeight {
        for cx in 0..<gridWidth {
          for b in 0..<boxesPerCell {

            // For the first bounding box (b=0) we have to read channels 0-24,
            // for b=1 we have to read channels 25-49, and so on.
            let channel = b*(numClasses + 5)

            // The slow way:
            /*
            let tx = features[[channel    , cy, cx] as [NSNumber]].floatValue
            let ty = features[[channel + 1, cy, cx] as [NSNumber]].floatValue
            let tw = features[[channel + 2, cy, cx] as [NSNumber]].floatValue
            let th = features[[channel + 3, cy, cx] as [NSNumber]].floatValue
            let tc = features[[channel + 4, cy, cx] as [NSNumber]].floatValue
            */

            // The fast way:
            let tx = Float(featurePointer[offset(channel    , cx, cy)])
            let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
            let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
            let th = Float(featurePointer[offset(channel + 3, cx, cy)])
            let tc = Float(featurePointer[offset(channel + 4, cx, cy)])

            // The predicted tx and ty coordinates are relative to the location
            // of the grid cell; we use the logistic sigmoid to constrain these
            // coordinates to the range 0 - 1. Then we add the cell coordinates
            // (0-12) and multiply by the number of pixels per grid cell (32).
            // Now x and y represent center of the bounding box in the original
            // 416x416 image space.
            let x = (Float(cx) + sigmoid(tx)) * blockSize
            let y = (Float(cy) + sigmoid(ty)) * blockSize

            // The size of the bounding box, tw and th, is predicted relative to
            // the size of an "anchor" box. Here we also transform the width and
            // height into the original 416x416 image space.
            let w = exp(tw) * anchors[2*b    ] * blockSize
            let h = exp(th) * anchors[2*b + 1] * blockSize

            // The confidence value for the bounding box is given by tc. We use
            // the logistic sigmoid to turn this into a percentage.
            let confidence = sigmoid(tc)

            // Gather the predicted classes for this anchor box and softmax them,
            // so we can interpret these numbers as percentages.
            var classes = [Float](repeating: 0, count: numClasses)
            for c in 0..<numClasses {
              // The slow way:
              //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
              // The fast way:
              classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
            }
            classes = softmax(classes)

            // Find the index of the class with the largest score.
            let (detectedClass, bestClassScore) = classes.argmax()

            // Combine the confidence score for the bounding box, which tells us
            // how likely it is that there is an object in this box (but not what
            // kind of object it is), with the largest class prediction, which
            // tells us what kind of object it detected (but not where).
            let confidenceInClass = bestClassScore * confidence

            // Since we compute 13x13x5 = 845 bounding boxes, we only want to
            // keep the ones whose combined score is over a certain threshold.
            if confidenceInClass > confidenceThreshold {
              let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                width: CGFloat(w), height: CGFloat(h))

              let prediction = Prediction(classIndex: detectedClass,
                                          score: confidenceInClass,
                                          rect: rect)
              predictions.append(prediction)
            }
          }
        }
      }
    }*/
}
