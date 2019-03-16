// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

var ui = ui || {}


const CONTROLS = [
  'up', 'down', 'left', 'right', 'leftclick', 'rightclick', 'scrollup',
  'scrolldown'
];
var controlsCaptured = [], labelsCaptured = [];

ui.init =
    function() {
  document.getElementById('user-help').style.visibility = 'visible';
  document.getElementById('user-help-text').innerText =
      'Add images to the classes below by clicking or holding';
  var controlButtons = document.getElementsByClassName('control-button');
  for (var i = 0; i < controlButtons.length; i++) {
    controlButtons[i].addEventListener('mouseover', function(event) {
      if (event.target.classList.contains('control-button')) {
        document.getElementById(event.target.id + '-icon').className =
            'control-icon center move-up';
        document.getElementById(event.target.id + '-add-icon').className =
            'add-icon';
      }
    });
    controlButtons[i].addEventListener('mouseout', function(event) {
      if (event.target.classList.contains('control-button')) {
        document.getElementById(event.target.id + '-icon').className =
            'control-icon center';
        document.getElementById(event.target.id + '-add-icon').className =
            'add-icon invisible';
      }
    });
  }
}



function
hideAllDropdowns() {
  let dropdownLists = document.getElementsByClassName('custom-option-list');
  for (var j = 0; j < dropdownLists.length; j++) {
    dropdownLists[j].className = 'custom-option-list hide';
  }
}

var customDropdowns = document.getElementsByClassName('custom-dropdown');
for (var i = 0; i < customDropdowns.length; i++) {
  customDropdowns[i].addEventListener('click', (event) => {
    hideAllDropdowns();
    const id = event.target.id + '-list';
    document.getElementById(id).className = 'custom-option-list';
  });
}

var customDropdownOptions = document.getElementsByClassName('custom-option');
for (var i = 0; i < customDropdownOptions.length; i++) {
  customDropdownOptions[i].addEventListener('click', (event) => {
    let dropdownID = event.target.parentNode.getAttribute('dropdownID');
    let dropdownList = document.getElementById(dropdownID + '-dropdown-list');
    dropdownList.getElementsByClassName('selected')[0].className =
        'custom-option';
    event.target.className = 'custom-option selected';
    event.target.parentNode.className = 'custom-option-list hide';
    document.getElementById(dropdownID).innerText = event.target.innerText;
  });
}

document.body.addEventListener('click', (event) => {
  if (event.target.className !== 'custom-dropdown') {
    hideAllDropdowns();
  }
})

const trainStatusElement = document.getElementById('train-status');
const downloadModel = document.getElementById('download-model');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
var selectedLearningRateValue =
    document.getElementById('learningRate-dropdown-list')
        .getElementsByClassName('selected')[0]
        .innerText;
learningRateElement.innerText = selectedLearningRateValue;

ui.getLearningRate = () => +learningRateElement.innerText;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
var batchSizeFractionValue =
    document.getElementById('batchSizeFraction-dropdown-list')
        .getElementsByClassName('selected')[0]
        .innerText;
batchSizeFractionElement.innerText = batchSizeFractionValue;

ui.getBatchSizeFraction = () => +batchSizeFractionElement.innerText;

const epochsElement = document.getElementById('epochs');
var epochsValue = document.getElementById('epochs-dropdown-list')
                      .getElementsByClassName('selected')[0]
                      .innerText;
epochsElement.innerText = epochsValue;

ui.getEpochs = () => +epochsElement.innerText;

const denseUnitsElement = document.getElementById('dense-units');
var denseUnitsValue = document.getElementById('dense-units-dropdown-list')
                          .getElementsByClassName('selected')[0]
                          .innerText;
denseUnitsElement.innerText = denseUnitsValue;

ui.getDenseUnits = () => +denseUnitsElement.innerText;

function removeActiveClass() {
  let activeElement = document.getElementsByClassName('active');
  while (activeElement.length > 0) {
    activeElement[0].className = 'control-inner-wrapper';
  }
}

ui.predictClass =
    function(classId) {
  removeActiveClass();
  classId = Math.floor(classId);
  document.getElementById(controlsCaptured[classId] + '-button').className =
      'control-inner-wrapper active';
  document.body.setAttribute('data-active', controlsCaptured[classId]);
}

    ui.isPredicting =
        function() {
  document.getElementById('predict').className = 'test-button hide';
  document.getElementById('webcam-outer-wrapper').style.border =
      '4px solid #00db8b';
  document.getElementById('stop-predict').className = 'stop-button';
  document.getElementById('bottom-section').style.pointerEvents = 'none';
  downloadModel.className = 'disabled';
};

ui.donePredicting =
            function() {
  document.getElementById('predict').className = 'test-button';
  document.getElementById('webcam-outer-wrapper').style.border =
      '2px solid #c8d0d8';
  document.getElementById('stop-predict').className = 'stop-button hide';
  document.getElementById('bottom-section').style.pointerEvents = 'all';
  downloadModel.className = '';
  removeActiveClass();
}

            ui.trainStatus =
                function(status) {
  trainStatusElement.innerText = status;
}

                ui.enableModelDownload =
                    function() {
  downloadModel.className = '';
}

                    ui.enableModelTest =
                        function() {
  document.getElementById('predict').className = 'test-button';
}

var addExampleHandler;

ui.setExampleHandler = function(handler) {
  addExampleHandler = handler;
};
let mouseDown = false;
const totals = [0, 0, 0, 0, 0, 0, 0, 0];

const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');
const leftClickButton = document.getElementById('leftclick');
const rightClickButton = document.getElementById('rightclick');
const scrollUpButton = document.getElementById('scrollup');
const scrollDownButton = document.getElementById('scrolldown');

const thumbDisplayed = {};
function timeout(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
ui.handler =
    async function(label) {
  mouseDown = true;
  const id = CONTROLS[label];
  const button = document.getElementById(id);
  const total = document.getElementById(id + '-total');

  document.body.removeAttribute('data-active');


  while (mouseDown) {
    totals[label] = totals[label] + 1;
    total.innerText = totals[label];
    for (var i = 0; i < totals.length; i++) {
      if (totals[i] > 0) {
        var isPresent = false;
        for (var j = 0; j < controlsCaptured.length; j++) {
          if (CONTROLS[i] === controlsCaptured[j]) {
            isPresent = true;
            break;
          }
        }
        if (!isPresent) {
          controlsCaptured.push(CONTROLS[i]);
          labelsCaptured.push(i);
          break;
        }
      }
    }
    addExampleHandler(label);
    await Promise.all([tf.nextFrame(), timeout(300)]);
  }
  document.body.setAttribute('data-active', CONTROLS[label]);
  if (controlsCaptured.length >= 2) {
    document.getElementById('train').className = 'train-button';
    ui.trainStatus('TRAIN');
    document.getElementById('predict').className = 'test-button disabled';
    downloadModel.className = 'disabled';
    document.getElementById('user-help-text').innerText =
        'Add more images or train the model';
  } else {
    document.getElementById('user-help-text').innerText =
        'Minimum of 2 classes required to train the model';
  }
}

    upButton.addEventListener('mousedown', () => ui.handler(0));
upButton.addEventListener('mouseup', () => {
  mouseDown = false;
});

downButton.addEventListener('mousedown', () => ui.handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => ui.handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => ui.handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

leftClickButton.addEventListener('mousedown', () => ui.handler(4));
leftClickButton.addEventListener('mouseup', () => mouseDown = false);

rightClickButton.addEventListener('mousedown', () => ui.handler(5));
rightClickButton.addEventListener('mouseup', () => mouseDown = false);

scrollUpButton.addEventListener('mousedown', () => ui.handler(6));
scrollUpButton.addEventListener('mouseup', () => mouseDown = false);

scrollDownButton.addEventListener('mousedown', () => ui.handler(7));
scrollDownButton.addEventListener('mouseup', () => mouseDown = false);

ui.drawThumb =
    function(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    thumbCanvas.style.display = 'block';
    document.getElementById(CONTROLS[label] + '-icon').style.top = '-50%';
    ui.draw(img, thumbCanvas);
  }
}

    ui.draw = function(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
