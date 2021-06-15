const formElement = document.getElementById('upload-form');
const responseContainer = document.getElementById('response-container');
const fileInput = document.getElementById('document');

formElement.addEventListener('submit', (e) => handleFormSubmit(e));

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
      responseContainer.innerHTML = JSON.stringify(data, null, 2);
      responseContainer.classList.remove("prettyprinted");
      console.log(data);
      PR.prettyPrint();
    });
}