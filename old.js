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
var trainCombinedImages = [];
testCombinedImages = [];

// Data parsing globals
var datasetLabelsTraining;
var datasetLabelsTesting;
var trainImages;
var testImages;
var shuffledTrainIndex = 0;
shuffledTestIndex = 0;
var trainIndices;
var testIndices;

// Call functions to parse the data
// Step 1: Mapping DONE
// Step 2: Build training images
// Step 3: Build test images

function createLogEntry(message) {
    var html = '<p>' + message + '</p>';
    $('#log').append(html);
}

function logCounter(message, counter)
{
    if(counter === 0)
    {
        $('#log').append('<p>' + message + ' <span id="counter">1</span></p>');
    }

    else
    {
        $('#counter').text(counter + 1);
    }
}

// Data is a string
function buildClasses(file) {
    // Log
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function (e) {
        var text = e.target.result;

        // Adding ', ' where there are spaces
        text = text.replace(/ /g, ', ');

        // building array
        var array = text.match(/\d+, \d+/g);

        for (var x = 0; x < array.length; x++) {
            // Getting label
            var label = array[x].match(/\d+/g).map(Number);

            // Pushing
            classes.push(label[1]);
            classLabels.push(label[0]);
        }

        createLogEntry("Finished parsing " + file.name);
    };
}

function buildTrainingData(file) {
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function (e) {
        var text = e.target.result;

        //console.log(text);

        // Building array
        text = text.match(/^.{5,}$/gm);

        // Looping through test images as CSV's
        for (var x = 0; x < text.length; x++) {
            // Splitting the text row, should be lenght 785
            var row = text[x].split(/,/g).map(Number);
            var label = row[0];
            row.shift();

            // The row doesn't have the class label anymore
            trainingCSV.push(row);
            trainingLabel.push(label);
            trainCombinedImages = trainCombinedImages.concat(row)
        }

        console.log(trainCombinedImages);

        // Setting constant
        number_training_set = trainingCSV.length;
        createLogEntry("Finished parsing " + file.name);

        dataParsing();
    };
}

function buildTrainingImages()
{
    createLogEntry("Building training images ...");

    const WIDTH = 784;
    const chunkSize = 4700;
    var images = [];
    var position = 0;

    // Setting datasetBytes buffer
    const datasetBytesBuffer = new ArrayBuffer(number_training_set * IMAGE_SIZE * 4);

    for(var i = 0; i < number_training_set / 4700; i++)
    {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        var imgData = ctx.createImageData(WIDTH, chunkSize);
        var data = imgData.data;
        var combindedImages = [];

        // Get a pointer to the current location in the image.
        var imageData = ctx.getImageData(0,0,WIDTH,chunkSize); //x,y,w,h
        
        // Wrap your array as a Uint8ClampedArray
        imageData.data.set(new Uint8ClampedArray(trainCombinedImages.slice(i, i * chunkSize))); // assuming values 0..255, RGBA, pre-mult.

        // Creating float32array
        const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);

        for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
        }

        // Repost the data.
        ctx.putImageData(imageData,0,0);

        logCounter("Training images chunks", i);
    }

    createLogEntry("Built training images");

    return new Float32Array(datasetBytesBuffer);
}

function buildTestingImages()
{
    createLogEntry("Building testing images ...");

    const WIDTH = 784;
    const chunkSize = 4700;
    var images = [];

    // Setting datasetBytes buffer
    const datasetBytesBuffer = new ArrayBuffer(number_training_set * IMAGE_SIZE * 4);

    for(var i = 0; i < number_training_set / 4700; i++)
    {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        var imgData = ctx.createImageData(WIDTH, chunkSize);
        var data = imgData.data;

        // Get a pointer to the current location in the image.
        var imageData = ctx.getImageData(0,0,WIDTH,chunkSize); //x,y,w,h
        
        // Wrap your array as a Uint8ClampedArray
        imageData.data.set(new Uint8ClampedArray(testCombinedImages.slice(i, i * chunkSize))); // assuming values 0..255, RGBA, pre-mult.

        // Creating float32array
        const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);

        for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
        }

        // Repost the data.
        ctx.putImageData(imageData,0,0);
        
        logCounter("Testing images chunks", i);
    }

    createLogEntry("Built testing images");

    return new Float32Array(datasetBytesBuffer);
}

function buildTestData(file) {
    createLogEntry("Parsing " + file.name + " ...");

    var reader = new FileReader();
    reader.readAsText(file);
    reader.onload = function (e) {
        var text = e.target.result;

        // Building array
        text = text.match(/^.{5,}$/gm);

        // Looping through test images as CSV's
        for (var x = 0; x < text.length; x++) {
            // Splitting the text row, should be lenght 785
            var row = text[x].split(/,/g).map(Number);
            var label = row[0];
            row.shift();

            // The row doesn't have the class label anymore
            testCSV.push(row);
            testLabel.push(label);
            testCombinedImages.concat(row);
        }

        // Setting constant
        number_testing_set = testCSV.length;
        createLogEntry("Finished parsing " + file.name);
    };
}

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 300;

async function train() {
    createLogEntry("Starting to train the model ...");

    // Looping through the training TRAIN_BATCHES of times
    for (let i = 0; i < TRAIN_BATCHES; i++) {
        console.log("Training ", i);
        const batch = tf.tidy(() => {
            const batch = nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;
        });

        await model.fit(
            batch.xs, batch.labels, { batchSize: BATCH_SIZE, epochs: 1 }
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }

    createLogEntry("Training complete");

    // Undisabling button
    document.getElementById('testButton').disabled = false;
}

function nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
    console.log(data[0]);
    // SOMETHING WRONG HERE
    for (let i = 0; i < batchSize; i++) {
        const idx = 2;

        const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
        batchImagesArray.set(image, i * IMAGE_SIZE);

        const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
        batchLabelsArray.set(label, i * NUM_CLASSES);

        //console.log(image);
        //console.log("Start ", idx * IMAGE_SIZE);
        //console.log("End ", idx * IMAGE_SIZE + IMAGE_SIZE);
    }
    //console.log(batchImagesArray);
    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
}

function nextTrainBatch(batchSize) {
    return nextBatch(
        batchSize, [trainImages, datasetLabelsTraining], () => {
            shuffledTrainIndex = (shuffledTrainIndex + 1) % trainIndices.length;
            return trainIndices[shuffledTrainIndex];
        });
}

function nextTestBatch(batchSize) {
    return nextBatch(batchSize, [testImages, datasetLabelsTesting], () => {
        shuffledTestIndex =
            (shuffledTestIndex + 1) % testIndices.length;
        return this.testIndices[shuffledTestIndex];
    });
}

function dataParsing()
{
    // Building test and training images as chunks
    trainImages = buildTrainingImages();
    testImages = buildTestingImages();
    console.log(trainImages);

    // Creating labels
    datasetLabelsTraining = new Uint8Array(trainingLabel);
    datasetLabelsTesting = new Uint8Array(testLabel);

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    trainIndices = tf.util.createShuffledIndices(number_training_set);
    testIndices = tf.util.createShuffledIndices(number_testing_set);

    createLog("Data loaded");

    //train();
}