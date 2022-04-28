//
//  ContentView.swift
//  TennisAR
//
//  Created by Nicholas Novelle on 26.04.2022.
//

import SwiftUI

struct ContentView: View {
    @State var elapsed = "waiting"
    var body: some View {
        VStack {
            Text(elapsed)
            Button("try again") {
                elapsed = test()
            }
        }.onLoad {
            elapsed = test()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

func test() -> String! {
    let image = UIImage.init(named: "test_image")
    
    let yolo = YOLO()
    
    let pixelBuffer = image!.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight)!
    
    return "your device completed the task in " + String(format: "%.0f", yolo.predict(image: pixelBuffer)*1000) + " ms"
}

