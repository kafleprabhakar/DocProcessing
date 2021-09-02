const formElement = document.getElementById('upload-form');
const responseContainer = document.getElementById('response-container');
const fileInput = document.getElementById('document');
const htmlForm = document.getElementById('parsed-form');
const submitBtn = document.getElementById('form-submit');
const imgContainer = document.getElementById('image-container');

formElement.addEventListener('submit', (e) => handleFormSubmit(e));


const templateAnnotation = { 
  "@context": "http://www.w3.org/ns/anno.jsonld",
  "id": "#annotation-1",
  "type": "Annotation",
  "body": [{
    "type": "TextualBody",
    "value": "Comments Here"
  }],
  "target": {
    "selector": [{
      "type": "FragmentSelector",
      "conformsTo": "http://www.w3.org/TR/media-frags/",
      "value": "xywh=pixel:0,0,200,400"
    }]
  }
}

function makeAnnotationOfCell(cell) {
  const annotations = [];
  for (var box of cell.boxes) {
    const annotation = JSON.parse(JSON.stringify(templateAnnotation));
    annotation.id = 'annotation-' + Math.random().toString(36).substring(7);
    annotation.body[0].value = cell.content;
    const x = box[0][0];
    const y = box[0][1];
    const w = box[1][0] - box[0][0];
    const h = box[1][1] - box[0][1];
    annotation.target.selector[0].value = `xywh=pixel:${x},${y},${w},${h}`;
    annotations.push(annotation);
  }
  return annotations;
}

function makeTableAnnotations(table) {
  const annotations = [];
  for (var row of table) {
    for (var cell of row) {
      annotations.push(...makeAnnotationOfCell(cell));
    }
  }
  return annotations;
}


function makeClusterElement() {
  const cluster = document.createElement('div');
  cluster.classList.add("cluster", "py-3", "px-2", "border", "border-primary", "rounded", "my-3");
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

function makeHTMLForm(clusters) {
  htmlForm.innerHTML = '';
  console.log("before initializing anno");
  var anno = Annotorious.init({
    image: 'processed-img',
    readOnly: true
  });
  console.log("after initializing anno");
  for (var cluster of clusters) {
    const clusterElement = makeClusterElement();
    if (cluster.type === 'checkboxes') {
      for (var element of cluster.data) {
        clusterElement.appendChild(makeCheckbox(element.label, element.percent_filled > 0.15));
      }
    } else if (cluster.type === 'uniform_table') {
      // clusterElement.append(json2Table(cluster.data));
      const thisTable = buildTable(cluster.data);
      const downloadableCsvLink = makeDownloadableCsv(cluster.data);
      clusterElement.innerHTML = thisTable;
      clusterElement.appendChild(downloadableCsvLink);
      
      const tableAnnotations = makeTableAnnotations(cluster.data);
      for (var annotation of tableAnnotations) {
        anno.addAnnotation(annotation);
      }
    }
    htmlForm.appendChild(clusterElement);
  }
}

function addImage(imgPath) {
  imgContainer.innerHTML = '';
  const image = document.createElement('img');
  image.setAttribute('src', imgPath);
  image.setAttribute('id', 'processed-img');
  image.classList.add('processed-image');
  imgContainer.appendChild(image);

  // var anno = Annotorious.init({
  //   image: 'processed-img',
  //   readOnly: true
  // });
  // anno.setAnnotations([templateAnnotation]);
  // anno.on('createAnnotation', function(annotation) {
  //   console.log('Created!');
  //   console.log(annotation);
  // });
}

function json2Table(json) {
  console.log('json');
  console.log(json);
  // https://dev.to/boxofcereal/how-to-generate-a-table-from-json-data-with-es6-methods-2eel
  let cols = Object.keys(json[0]);
  //Map over columns, make headers, join into string
  let headerRow = cols
    .map(col => `<th>${col}</th>`)
    .join("");
  
  let rows = json
    .map(row => {
      let tds = cols.map(col => `<td>${row[col]}</td>`).join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  //build the table
  const table = `
	<table class="table table-striped table-bordered">
		<thead>
			<tr>${headerRow}</tr>
		<thead>
		<tbody>
			${rows}
		<tbody>
	<table>`;

  return table;
}

function buildTable(table) {
  let rows = table.map(row => {
    let tds = row.map(cell => `<td>${cell.content}</td>`).join("");
    return `<tr>${tds}</tr>`;
  }).join("");
  const htmlTable = `
  <table class="table table-striped table-bordered">
    <tbody>
      ${rows}
    </tbody>
  </table>`;
  return htmlTable;
}


function makeDownloadableCsv(table) {
  const csv = table.map(row => {
    let tds = row.map(cell => `${cell.content}`).join(",");
    return `${tds}\n`;
  }).join("");
  const csvBlob = new Blob([csv], {type: "text/csv;charset=utf-8"});
  const csvUrl = URL.createObjectURL(csvBlob);
  const csvLink = document.createElement('a');
  csvLink.setAttribute('href', csvUrl);
  csvLink.setAttribute('download', 'table.csv');
  csvLink.innerHTML = 'Download CSV';
  csvLink.classList.add('btn', 'btn-outline-primary');
  return csvLink;
}


function handleFormSubmit(e) {
  e.preventDefault();

  const toSend = new FormData(formElement);
  toSend.append('document', fileInput.files[0]);
  
  submitBtn.classList.add('disabled');
  submitBtn.disabled = true;
  
  fetch(formElement.action, {
    method:'POST',
    body: toSend,
  }).then(response => response.json())
    .then(data => {
      // responseContainer.innerHTML = JSON.stringify(data, null, 2);
      addImage(data.image);
      makeHTMLForm(data.clusters);
      console.log('data clusters');
      console.log(data.clusters);
    })
    .finally(() => {
      submitBtn.disabled = false;
      submitBtn.classList.remove('disabled');
    });
  
}


