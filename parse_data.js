var classes = [], classLabels = [];
var number_testing_set, number_training_set;
var trainCSV = [], trainLabel = [], trainChunk = [], trainingImages, trainingLabel;
var testCSV = [], testLabel = [], testChunk = [], testingImages, testingLabel;
const IMAGE_SIZE = 784, NUM_CLASSES = 47, chunkSize = 4700; // 4700 divides evenly in both sets
let trainIndicies, testIndicies;

function createLogEntry(message) {
    var html = '<p>' + message + '</p>';
    $('#log').append(html);
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

function parseTrain(file)
{
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
            trainCSV.push(row);
            trainLabel.push(label);
        }

        // Setting constant
        number_training_set = trainCSV.length;

        /*******************************************************************************************************************/
        var index = 0;
        trainingImages = new Float32Array(IMAGE_SIZE * number_training_set);
        
        for(let i = 0; i < trainCSV.length; i++)
        {
            for(let x = 0; x < trainCSV[i].length; x++)
            {
                trainingImages[index] = trainCSV[i][x] / 255;
                index++;
            }
        }

        // Check if image is empty
        for(let i = 0; i < trainingImages.length / IMAGE_SIZE; i++)
        {
            if(Math.max.apply(Math, trainingImages.slice(i * IMAGE_SIZE, i * IMAGE_SIZE + IMAGE_SIZE)) === 0)
            {
                console.log("Image " + i + " is empty!");
                //console.log(testingImages.slice(i * IMAGE_SIZE, i * IMAGE_SIZE + IMAGE_SIZE));
            }
        }

        // Time to do the labels, we want them in a Uint8Array
        const labelBuffer = new ArrayBuffer(number_training_set * NUM_CLASSES);
        trainingLabel = new Uint8Array(labelBuffer);
        index = 0;

        for(let i = 0; i < trainLabel.length; i++)
        {
            let min = i * NUM_CLASSES;
            let max = min + NUM_CLASSES;
            let index = min + trainLabel[i];

            // Set this to the Uint8Array
            trainingLabel[index] = 1;

            // console.log("Training Label: ", trainLabel[i]);
            // console.log(trainingLabel.slice(min, max));
        }

        // Creating shuffled indicies for the training set, this will be used
        // when we select a random dataset element for training
        trainIndicies = tf.util.createShuffledIndices(number_training_set);

        // Parse training set finishes last always, start training
        train();
        /*******************************************************************************************************************/

        createLogEntry("Finished parsing " + file.name);
    };
}

async function parseTest(file) {
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
            //testCombinedImages.concat(row);
        }

        // Setting constant
        number_testing_set = testCSV.length;
        
        //console.log(testCSV);

        /*******************************************************************************************************************/
        var index = 0;
        testingImages = new Float32Array(IMAGE_SIZE * number_testing_set);
        
        for(let i = 0; i < testCSV.length; i++)
        {
            for(let x = 0; x < testCSV[i].length; x++)
            {
                testingImages[index] = testCSV[i][x] / 255;
                index++;
            }
        }

        // Check if image is empty
        for(let i = 0; i < testingImages.length / IMAGE_SIZE; i++)
        {
            if(Math.max.apply(Math, testingImages.slice(i * IMAGE_SIZE, i * IMAGE_SIZE + IMAGE_SIZE)) === 0)
            {
                console.log("Image " + i + " is empty!");
                //console.log(testingImages.slice(i * IMAGE_SIZE, i * IMAGE_SIZE + IMAGE_SIZE));
            }
        }

        // Time to do the labels, we want them in a Uint8Array
        const labelBuffer = new ArrayBuffer(number_testing_set * NUM_CLASSES);
        testingLabel = new Uint8Array(labelBuffer);
        index = 0;

        for(let i = 0; i < testLabel.length; i++)
        {
            let min = i * NUM_CLASSES;
            let max = min + NUM_CLASSES;
            let index = min + testLabel[i];

            // Set this to the Uint8Array
            testingLabel[index] = 1;
        }

        // Creating shuffled indicies for the training set, this will be used
        // when we select a random dataset element for training
        testIndicies = tf.util.createShuffledIndices(number_testing_set);
        /*******************************************************************************************************************/
        
        createLogEntry("Finished parsing " + file.name);
    };
}

const BATCH_SIZE = 100;
let trainPosition = 0;

function nextTrainBatch()
{
    // Defining array for the images and classes
    const batchImagesArray = new Float32Array(BATCH_SIZE * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(BATCH_SIZE * NUM_CLASSES);

    for(let i = 0; i < BATCH_SIZE; i++)
    {
        const image = trainingImages.slice(trainPosition * IMAGE_SIZE, trainPosition * IMAGE_SIZE + IMAGE_SIZE);
        const label = trainingLabel.slice(trainPosition * NUM_CLASSES, trainPosition * NUM_CLASSES + NUM_CLASSES);

        // Setting arrays to values
        batchImagesArray.set(image, i * IMAGE_SIZE);
        batchLabelsArray.set(label, i * NUM_CLASSES);
        trainPosition++;
    }

    //console.log(batchImagesArray);
    //console.log(batchLabelsArray);

    const xs = tf.tensor2d(batchImagesArray, [BATCH_SIZE, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [BATCH_SIZE, NUM_CLASSES]);

    return {xs, labels};
}

function nextTestBatch()
{
    let choice = Math.floor(Math.random() * testCSV.length);

    // Defining array for the images and classes
    const batchImagesArray = new Float32Array(1 * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(1 * NUM_CLASSES);

    for(let i = 0; i < 1; i++)
    {
        const image = testingImages.slice(choice * IMAGE_SIZE, choice * IMAGE_SIZE + IMAGE_SIZE);
        const label = testingLabel.slice(choice * NUM_CLASSES, choice * NUM_CLASSES + NUM_CLASSES);

        // Setting arrays to values
        batchImagesArray.set(image, i * IMAGE_SIZE);
        batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    console.log(batchImagesArray);
    console.log(batchLabelsArray);

    const xs = tf.tensor2d(batchImagesArray, [1, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [1, NUM_CLASSES]);

    return {xs, labels};
}

async function train() {
    createLogEntry("Starting to train the model ...");

    // Train on entire dataset
    for(let i = 0; i < number_training_set / BATCH_SIZE; i++)
    {
        console.log("Train: ", i);
        // Getting the batch
        const batch = tf.tidy(() => {
            const batch = nextTrainBatch();
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;
        });

        console.log(batch);

        await model.fit(
            batch.xs, batch.labels, {batchSize: BATCH_SIZE, epochs: 1}
        );

        tf.dispose(batch);

        await tf.nextFrame();
    }

    createLogEntry("Training complete");

    // Undisabling button
    document.getElementById('testButton').disabled = false;
}

function Flaot32Concat(first, second)
{
    var firstLength = first.length,
        result = new Float32Array(firstLength + second.length);

    result.set(first);
    result.set(second, firstLength);

    return result;
}