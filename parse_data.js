var classes = [];
var classLabels = [];
var images = [];
var trainingLabel = [];
var trainingCSV = [];
var testLabel = [];
var testCSV = [];

// Image information
const IMAGE_SIZE = 784;
const NUM_CLASSES = 47;
var number_training_set;
var number_testing_set;
var number_dataset;

// Call functions to parse the data
        // Step 1: Mapping DONE
        // Step 2: Build training images
        // Step 3: Build test images

function createLogEntry(message) {
    var html = '<p>' + message + '</p>';
    $('#log').append(html);
}
        
// Data is a string
function buildClasses(file)
{
    // Log
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function(e) {
        var text = e.target.result;
    
        // Adding ', ' where there are spaces
        text = text.replace(/ /g, ', ');

        // building array
        var array = text.match(/\d+, \d+/g);

        for(var x = 0; x < array.length; x++)
        {
            // Getting label
            var label = array[x].match(/\d+/g).map(Number);

            // Pushing
            classes.push(label[1]);
            classLabels.push(label[0]);
        }

        createLogEntry("Finished parsing " + file.name);
    };
}

function buildTrainingData(file)
{
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function(e) {
        var text = e.target.result;
    
        //console.log(text);

        // Building array
        text = text.match(/^.{5,}$/gm);
        
        // Looping through test images as CSV's
        for(var x = 0; x < text.length; x++)
        {
            // Splitting the text row, should be lenght 785
            var row = text[x].split(/,/g).map(Number);
            var label = row[0];
            row.shift();
            
            // The row doesn't have the class label anymore
            trainingCSV.push(row);
            trainingLabel.push(label);
        }
        
        // Setting constant
        number_training_set = trainingCSV.length;
        //buildImages();
        createLogEntry("Finished parsing " + file.name);

        loadData();
    };
}

function buildTestData(file)
{
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function(e) {
        var text = e.target.result;

        // Building array
        text = text.match(/^.{5,}$/gm);
        
        // Looping through test images as CSV's
        for(var x = 0; x < text.length; x++)
        {
            // Splitting the text row, should be lenght 785
            var row = text[x].split(/,/g).map(Number);
            var label = row[0];
            row.shift();
            
            // The row doesn't have the class label anymore
            testCSV.push(row);
            testLabel.push(label);
        }
        
        // Setting constant
        number_testing_set = testCSV.length;
        createLogEntry("Finished parsing " + file.name);
    };
}

function isPostive(element, index, array) {
    return element > 0;
  }

function loadData()
{
    createLog("Building images ...");
    console.log("test ", number_testing_set);
    console.log("training ", number_training_set);

    number_dataset = number_testing_set + number_training_set;

    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Setting image constants
    img.width = IMAGE_SIZE;
    img.height = number_dataset;

    // Setting datasetBytes buffer
    const datasetBytesBuffer =
            new ArrayBuffer(number_dataset * IMAGE_SIZE * 4);

    const chunkSize = 4700;
    canvas.width = img.width;
    canvas.height = chunkSize;

    for (let i = 0; i < number_dataset / chunkSize; i++)
    {
        // Creating float32array
        const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);

        ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let j = 0; j < imageData.data.length / 4; j++) 
        {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
    }

    // Creating labels
    var datasetLabelsTraining = new Uint8Array(trainingLabel);
    var datasetLabelsTesting = new Uint8Array(testLabel);

    //console.log(datasetBytesBuffer);

    var datasetImages = new Float32Array(datasetBytesBuffer);
    
    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    var trainIndices = tf.util.createShuffledIndices(number_training_set);
    var testIndices = tf.util.createShuffledIndices(number_testing_set);

    // Slice the the images and labels into train and test sets.
    var trainImages = datasetImages.slice(0, IMAGE_SIZE * number_training_set);
    var testImages = datasetImages.slice(IMAGE_SIZE * number_training_set);

    //console.log(trainIndices);
    //console.log(testImages);
    console.log(datasetLabelsTraining);
    createLog("Images built");
}

function nextTrainBatch(batchSize)
{
    
}