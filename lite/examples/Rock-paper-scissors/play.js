// Element
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const vid = document.getElementById("computer_video");
const restartButton = document.getElementById("restart")
const showHandView = document.getElementById("show_hand")
const uResult = document.getElementById("you_result")
const cResult = document.getElementById("computer_result")
const videoContent = document.getElementById("video_content")
const resultLb = document.getElementById("result")
const nexRound = document.getElementById("next_round")

// Gane state
const loadModel = 0
const waitHands = 1
const ready = 2
const countDown = 2
const videoPlay = 3
const checkResult = 4
const finishRound = 5
const showResult = 6

var state = loadModel

// Variable
var comWins = 0
var youWins = 0
var isStart = false
var frameNoHands = 0
var isHands = false
var isWaiting = true
var isCoundown = false
var isCheckResult = false
var isFirstHadHands = false
var tfliteModel = null
var finish = false
var isStop = false
var comReult = 0

var youResult = -1
var lastImageGetResult = null

async function loadTfliteModel() {
  tfliteModel = await tflite.loadTFLiteModel('src/hand_classifier.tflite');
  console.log("finish load model")
}

loadTfliteModel()

function startGame() {
  if (state == countDown) { return }
  state = countDown
  var countDownTime = 3
  var x = setInterval(function() {
  if (countDownTime == 0) {
    clearInterval(x);
    document.getElementById("waiting").innerHTML = "";
    countDownTime = 3
    if (!isHands) {
      state = waitHands
      showHandView.style.display = "initial"
    } else {
      startRound();
    }
  } else {
    document.getElementById("waiting").innerHTML = countDownTime.toString();
    countDownTime -= 1
  }
  }, 500);
}

function startRound() {
  if (state == videoPlay) { return }
  state = videoPlay
  comReult = Math.floor(Math.random() * 3);
  switch (comReult) {
    case 0:
      videoContent.innerHTML = '<video id="computer-video"><source src="src/paper1.mp4" type="video/mp4" /></video>'
      break
    case 1:
    videoContent.innerHTML = '<video id="computer-video"><source src="src/rock1.mp4" type="video/mp4" /></video>'
      break
    case 2:
    videoContent.innerHTML = '<video id="computer-video"><source src="src/scissor1.mp4" type="video/mp4" /></video>'
      break
  }

  const vid = document.getElementById("computer-video")
  vid.play()
  vid.onplaying = function() {
    setTimeout(function(){
      state = checkResult
      setTimeout(function() {
        state = finishRound
      }, 400)
    }, 2600);
  }
}

function restart() {
  youWins = 0
  comWins = 0
  frameNoHands = 0
  cResult.innerText = comWins.toString() + " wins"
  uResult.innerText = youWins.toString() + " wins"
  state = waitHands
  showHandView.style.display = "initial"
  restartButton.style.display = "none"

}

const hands = new Hands({locateFile: (file) => {
  console.log(file)
  console.log(Date())
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

function onResults(results) {
  if (state == showResult) { return }
  if (state == finishRound) {
    state = showResult
    camera.stop()
    if (youResult != -1) {
      canvasCtx.save();
      canvasCtx.translate(canvasElement.width, 0);
      canvasCtx.scale(-1, 1);
      canvasCtx.drawImage(
        lastImageGetResult, 0, 0, canvasElement.width, canvasElement.height);
      var isWin = false
      if (comReult == youResult) {
        resultLb.innerText = "You drew"
      } else 
      if ((comReult == 0 && youResult == 1)
        || (comReult == 1 && youResult == 2)
        || (comReult == 2 && youResult == 0)) {
          resultLb.innerText = "Computer win"
          comWins += 1
          cResult.innerText = comWins.toString() + " wins"
        } else {
          resultLb.innerText = "You win"
          youWins += 1
          uResult.innerText = youWins.toString() + " wins"
        }
        youResult = -1
    } else {
      resultLb.innerText = "Can not detect hand"
    }
    resultLb.style.display = "initial"
    nexRound.style.display = "initial"
    setTimeout(function() {
      resultLb.style.display = "none"
      nexRound.style.display = "none"
      camera.start()
      state = waitHands
      videoContent.innerHTML = '<p style="font-size: 200px;">?</p>'
    }, 3000)
    return;
  }
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks.length > 0) {
    isHands = true
    frameNoHands = 0
    const landmarks = results.multiHandLandmarks[0]
    drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                        {color: '#00FF00', lineWidth: 2});
    if (state == checkResult) {
      var landmarksResize = []
      for (const landmark of landmarks) {
        landmarksResize.push(landmark["x"])
        landmarksResize.push(landmark["y"])
        landmarksResize.push(landmark["z"])
      }
      const input = [landmarksResize]
      const inputTensor = tf.tensor(input)
      let outputTensor = tfliteModel.predict(inputTensor);
      let output = outputTensor.dataSync()
      const max = Math.max(...output);
      if (max > 0.8) {
        youResult = output.indexOf(max);
        lastImageGetResult = results.image.cloneNode()
        console.log(youResult);
      }
    } else if (state == waitHands) {
      startGame()
      showHandView.style.display = "none"
      restartButton.style.display = "none"
    }
  } else {
    isHands = false
    if (state == waitHands) {
      frameNoHands += 1
      if (frameNoHands == 50) {
        showHandView.style.display = "initial"
      }
      if (frameNoHands == 150) {
        if (comWins > 0 || youWins > 0) {
          restartButton.style.display = "initial"
        }
      }
    }
  }
  canvasCtx.restore();
}

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
    if (state == loadModel) {
      state = waitHands
      showHandView.style.display = "initial"
    }
  },
  width: 1000,
  height: 720
});
camera.start();
