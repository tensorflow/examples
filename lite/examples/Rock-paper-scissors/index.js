function play() {
  window.location.href="/play.html"
}

const videoElement = document.getElementsByClassName('input-video')[0];
const canvasElement = document.getElementsByClassName('output-canvas')[0];
const canvasCtx = canvasElement.getContext('2d');


function onResults(results) {
  camera.stop()
  document.getElementById("beginbtn").style.display = "initial"
  document.getElementById("loading").innerText = "File download completed"
  document.getElementById("loadView").style.display = "none"
}

const hands = new Hands({locateFile: (file) => {
  document.getElementById("loading").innerText = "Loading file " + file
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1000,
  height: 720
});
camera.start();
