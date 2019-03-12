var model;
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
        units: 10,
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

function startTraining()
{
    
}

// When files are uploaded, this function triggers
$( "#trainingFiles" ).change(function() {
    // We need 3 files: emnist-balanced-mapping, emnist-balanced-train, emnist-balanced-test
    //get file object
    var file = document.getElementById('trainingFiles').files;
    var flag = 1;

    if(file.length === 3)
    {
        for(var i = 0; i < file.length; i++)
        {
            // Making sure the files are the correct one
            if (file[i] && checkFileName(file[i].name) !== -1) {

                // create reader
                console.log(file[i]);
                var reader = new FileReader();
                reader.readAsText(file[i]);
                reader.onload = function(e) {

                    // browser completed reading file - display it
                    alert(e.target.result);

                    // Call functions to parse the data
                    // Step 1: Mapping
                    // Step 2: Build training images
                    // Step 3: Build test images
                };
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
        // Add error messages
    }

    if(flag)
    {
        // start parsing files
        createAlert("All files were accepted!", "success");
        createLog("Start parsing files...");
    }
});

function main()
{
    // Creating tensorflow model for training
    createModel();

    console.log(model);
    
    $('#cardFooter').show('cardFooter');
}

main();