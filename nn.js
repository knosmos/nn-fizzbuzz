// set up model
const model = tf.sequential(
    {
        layers: [
            tf.layers.dense({
                inputShape: [10], units: 100, activation: 'relu',
                weights: [tf.randomUniform([10, 100], -1, 1), tf.randomUniform([100], -1, 1)]
            }),
            tf.layers.dense({
                inputShape: [100], units: 100, activation: 'relu',
                weights: [tf.randomUniform([100, 100], -1, 1), tf.randomUniform([100], -1, 1)]
            }),
            tf.layers.dense({inputShape: [100], units: 4, activation: 'sigmoid'})
        ]
    }
)

model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

// generate training data (train on numbers from 100-1023)
let x = [];
let y = [];
for (let i = 100; i < 1024; i++) {
    // get binary representation
    let a = [];
    for (let j = 0; j < 10; j++) {
        a.push((i >> j) & 1);
    }
    x.push(a);
    // solve fizzbuzz
    if (i % 15 == 0) {
        y.push([0, 0, 0, 1]);
    }
    else if (i % 5 == 0) {
        y.push([0, 0, 1, 0]);
    }
    else if (i % 3 == 0) {
        y.push([0, 1, 0, 0]);
    }
    else {
        y.push([1, 0, 0, 0]);
    }
}

// train model
function onEpochEnd(epoch, logs) {
    document.getElementById("status").innerHTML = `epoch ${epoch}: ${JSON.stringify(logs)}`;
}

model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 1000,
    batchSize: 512,
    callbacks: {onEpochEnd}
}).then(info => {
    document.getElementById("status").innerHTML = "training complete";

    // test model
    for (let i = 1; i < 101; i++) {
        let a = [];
        for (let j = 0; j < 10; j++) {
            a.push((i >> j) & 1);
        }
        let output = model.predict(tf.tensor([a])).dataSync();
        let outputMax = output.indexOf(Math.max(...output));
        switch (outputMax) {
            case 0:
                res = i;
                break;
            case 1:
                res = "Fizz";
                break;
            case 2:
                res = "Buzz";
                break;
            case 3:
                res = "FizzBuzz";
                break;
        }
        console.log(i, res);
        document.getElementById("results").innerHTML += `${i}: ${res}<br>`;
    }    
});