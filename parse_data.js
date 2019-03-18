var classes = [], classLabels = [];
var number_testing_set, number_training_set;
var trainCSV = [], trainLabel = [];
var testCSV = [], testLabel = [];
const IMAGE_SIZE = 784, NUM_CLASSES = 47;

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
        //text = text.match(/^.{5,}$/gm);
        text = text.match(/^.{5,}$/gm);
        console.log(text);
        var combinedArray = [];
        // Looping through test images as CSV's
        for (var x = 0; x < text.length; x++) {
            // Splitting the text row, should be lenght 785
            var row = text[x].split(/,/g).map(Number);
            var label = row[0];
            row.shift();

            // The row doesn't have the class label anymore
            //trainCSV.push(row);
            combinedArray = combinedArray.concat(row);
            trainLabel.push(label);
            //testCombinedImages.concat(row);
        }
        console.log(combinedArray);
        
        // Setting constant
        number_training_set = trainCSV.length;
        createLogEntry("Finished parsing " + file.name);
    };
}

function parseTest(file) {
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
        createLogEntry("Finished parsing " + file.name);
    };
}

const BATCH_SIZE = 100;
var position = 0;

async function train() {
    createLogEntry("Starting to train the model ...");

    // Looping through the training TRAIN_BATCHES of times
    for (let i = 0; i < number_training_set / BATCH_SIZE; i++) {
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