// set up model
INPUT_SIZE = 16;
const model = tf.sequential(
    {
        layers: [
            tf.layers.dense({
                inputShape: [INPUT_SIZE], units: 100, activation: 'relu',
                weights: [tf.randomUniform([INPUT_SIZE, 100], -1, 1), tf.randomUniform([100], -1, 1)]
            }),
            tf.layers.dense({
                inputShape: [100], units: 50, activation: 'relu',
                weights: [tf.randomUniform([100, 50], -1, 1), tf.randomUniform([50], -1, 1)]
            }),
            tf.layers.dense({inputShape: [50], units: 4, activation: 'softmax'})
        ]
    }
)

model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

// solve fizzbuzz
function solve(i) {
    if (i % 15 == 0) return 3;
    if (i % 5 == 0) return 2;
    if (i % 3 == 0) return 1;
    return 0;
}

// generate training data (train on numbers from 100-1023)
let x = [];
let y = [];
for (let i = 101; i < 4096; i++) {
    // get binary representation
    let a = [];
    for (let j = 0; j < INPUT_SIZE; j++) {
        a.push((i >> j) & 1);
    }
    x.push(a);
    // get fizzbuzz result
    let r = [0, 0, 0, 0];
    r[solve(i)] = 1;
    y.push(r);
}

// train model
function onEpochEnd(epoch, logs) {
    document.getElementById("epoch").innerHTML = (epoch + 1) + "/1000";
    document.getElementById("loss").innerHTML = logs.loss;
    document.getElementById("accuracy").innerHTML = logs.acc;

    console.log(`epoch ${epoch + 1}/1000: ${logs.loss}, ${logs.acc}`);
}

model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 1000,
    batchSize: 256,
    callbacks: {onEpochEnd}
}).then(info => {
    // test model
    let correct = 0;
    for (let i = 1; i < 101; i++) {
        let a = [];
        for (let j = 0; j < INPUT_SIZE; j++) {
            a.push((i >> j) & 1);
        }
        let output = model.predict(tf.tensor([a])).dataSync();
        let outputMax = output.indexOf(Math.max(...output));
        let res = [i, "Fizz", "Buzz", "FizzBuzz"][outputMax];

        // calculate correctness
        let pred_correct = outputMax == solve(i);
        if (pred_correct) {
            correct++;
        }
        
        // output
        console.log(i, res);
        document.getElementById("results").innerHTML += `<span class=${[pred_correct ? "correct" : "incorrect"]}>${res}</span>${i < 100 ? ", " : ""}`;
    }
    document.getElementById("inference-text").style.display = "block";
    document.getElementById("status").innerHTML = `training complete: ${correct}/100 correct`;
    document.getElementById("results").style.opacity = "1";
    document.getElementById("tester").style.display = "contents";
});

// interactive testing
function testNumber() {
    let num = parseInt(document.getElementById("number").value);
    let a = [];
    for (let j = 0; j < INPUT_SIZE; j++) {
        a.push((num >> j) & 1);
    }

    let output = model.predict(tf.tensor([a])).dataSync();
    console.log(output);
    let outputMax = output.indexOf(Math.max(...output));
    let res = [num, "Fizz", "Buzz", "FizzBuzz"][outputMax];

    let pred_correct = outputMax == solve(num);
    document.getElementById("test-inference").style.opacity = 1;
    document.getElementById("test-inference").innerHTML = `Inference result: <span class=${pred_correct ? "correct" : "incorrect"}>${res}</span> (${pred_correct ? "correct" : "incorrect"})
Model output probabilities: [${Array.from(output).map(x => parseFloat(x).toFixed(2)).join(", ")}]`;
}