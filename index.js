var model;
var success = 0;
var total = 0;
const fileNames = ["emnist-balanced-mapping.txt", "emnist-balanced-test.csv", "emnist-balanced-train.csv"];

function checkFileName(fileName)
{
    // Returns -1 if not found
    return fileNames.indexOf(fileName);
}

function createLog(message)
{
    var html = '<p>' + message + '</p>';
    $('#log').append(html);
}

function createAlert(message, type)
{
    var html = '<div class="alert alert-' + type + ' alert-dismissible fade show" role="alert">'
    + message +
    '<button type="button" class="close" data-dismiss="alert" aria-label="Close">' +
      '<span aria-hidden="true">&times;</span>' +
    '</button>' +
'</div>';

    $('#response').append(html);
}

function createModel()
{
    createLog('Create model ...');
    model = tf.sequential();
    createLog('Model created');

    createLog('Add layers ...');
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2,2],
        strides: [2,2]
    }));

    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2,2],
        strides: [2,2]
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
    }));

    createLog('Layers created');

    createLog('Start compiling ...');

    model.compile({
        optimizer: tf.train.sgd(0.15),
        loss: 'categoricalCrossentropy'
    });
    createLog('Compiled');
}

// When files are uploaded, this function triggers
$( "#trainingFiles" ).change(function() {
    // We need 3 files: emnist-balanced-mapping, emnist-balanced-train, emnist-balanced-test
    //get file object
    var file = document.getElementById('trainingFiles').files;
    var flag = 1;
    createLog("Start parsing files ...");

    if(file.length === 3)
    {
        var pass = 1;

        for(var i = 0; i < file.length; i++)
        {
            // Making sure the files are the correct one
            if (file[i] && checkFileName(file[i].name) !== -1) 
            {
                // Switching the file names
                switch(file[i].name)
                {
                    // Mapping
                    case fileNames[0]:
                        buildClasses(file[i]);
                        break;

                    // Test
                    case fileNames[1]:
                        parseTest(file[i]);
                        break;

                    // Train
                    case fileNames[2]:
                        parseTrain(file[i]);
                        break;

                    default:
                        createAlert("<strong>Error:</strong> Unknown file name. Please upload the correct files.", "danger");
                        createLog("Please upload the correct files.");
                        pass = 0;
                        break;
                }

                if(pass === 0)
                {
                    break;
                }
            }

            else
            {
                flag = 0;
                createAlert("File <strong>" + file[i].name + "</strong> is not accepted. Please try again.", "danger");
                createLog("Files were rejected. Please try again.");
                break;
            }
        }
    }
    
    else
    {
        flag = 0;
        createAlert("<strong>Error:</strong> Not enough files. We need three files and they are: " + fileNames, "danger");
        // Add error messages
    }

    if(flag)
    {
        // Files were accepted
    }
});

function main()
{
    // Creating tensorflow model for training
    createModel();

    console.log(model);
    
    $('#cardFooter').show('cardFooter');

    // Adding button
    // Appending the a button for testing
    $("#test").append('<button class="btn btn-primary" id="testButton" disabled>Test Model</button>');

    document.getElementById('testButton').addEventListener('click', async (el,ev) => {
        /*for(var i = 0; i < 15000; i++)
        {
            const batch = nextTestBatch();
            await predict(batch);
        }

        console.log("Number of images tested with: ", total);
        console.log("Number of test predicted successfully: ", success);
        console.log("Accuracy: " + ((success / total) * 100) + "%");
        total = 0;
        success = 0;*/
        const batch = nextTestBatch();
        await predict(batch);
    });
}

async function predict(batch) {
    tf.tidy(() => {
        console.log(batch);
        const input_value = Array.from(batch.labels.argMax(1).dataSync());

        const div = document.createElement('div');
        div.className = 'prediction-div';

        const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

        const prediction_value = Array.from(output.argMax(1).dataSync());
        const image = batch.xs.slice([0, 0], [1, batch.xs.shape[1]]);
        console.log(image);

        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);

        const label = document.createElement('div');
        label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;

        if (prediction_value - input_value == 0) {
            label.innerHTML += '<br>Value recognized successfully';
        } else {
            label.innerHTML += '<br>Recognition failed!'
        }

        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('test').appendChild(div);
        total++;

        if(prediction_value - input_value == 0) {
            success++;
        }
    });
}

function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; i++) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

main();