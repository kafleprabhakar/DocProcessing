const formElement = document.getElementById('upload-form');
const responseContainer = document.getElementById('response-container');
const fileInput = document.getElementById('document');
const htmlForm = document.getElementById('parsed-form');
const imgContainer = document.getElementById('image-container');

formElement.addEventListener('submit', (e) => handleFormSubmit(e));

function makeClusterElement() {
  const cluster = document.createElement('div');
  cluster.classList.add("cluster", "py-3", "px-2", "border", "rounded", "my-3");
  return cluster
}

function makeCheckbox(label, status) {
  const checkbox = document.getElementById('checkbox-template').content.cloneNode(true);
  const inputLabel = checkbox.querySelector('.form-check-label');
  const input = checkbox.querySelector('.form-check-input');
  // console.log(checkbox);
  input.checked = status;
  inputLabel.textContent = label;
  const randomId = 'checkbox_' + Math.random().toString(36).substring(7);
  input.setAttribute('id', randomId);
  inputLabel.setAttribute('for', randomId);
  // label.for = randomId;
  return checkbox;
}

function makeHTMLForm(data) {
  htmlForm.innerHTML = '';
  for (var cluster of data) {
    // console.log(cluster);
    // const clusterElement = document.createElement('div');
    // const clusterElement = document.getElementById('cluster-template').content.cloneNode(true);
    const clusterElement = makeClusterElement();
    for (var element of cluster) {
      clusterElement.appendChild(makeCheckbox(element.label, element.percent_filled > 0.15));
    }
    htmlForm.appendChild(clusterElement);
  }
}

function addImage(imgPath) {
  imgContainer.innerHTML = '';
  const image = document.createElement('img');
  image.setAttribute('src', imgPath);
  image.classList.add('processed-image');
  imgContainer.appendChild(image);
}

function handleFormSubmit(e) {
  e.preventDefault();
  console.log(e);
  const toSend = new FormData(formElement);
  toSend.append('document', fileInput.files[0])
  console.log('to send', ...toSend);
  console.log(fileInput.files);
  fetch(formElement.action, {
    method:'POST',
    body: toSend,
  }).then(response => response.json())
    .then(data => {
      // responseContainer.innerHTML = JSON.stringify(data, null, 2);
      makeHTMLForm(data.clusters);
      addImage(data.image);
    });
}