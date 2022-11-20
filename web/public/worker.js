importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");
import("./ai/renjuai.js")
    .then((module) => {
        module.default()
            .then(async () => {
                console.log(await module.test('x'));
            })
            .catch((err) => {
                console.log('Unable to load AI module', err);
            });
    })
    .catch((err) => {
        console.log('Unable to load renjuai.js', err);
    });

function predict(state_tensor) {
    return {
        probability_matrix: state_tensor[0],
        score: 0.3,
    };
}

function console_log(text) {
    console.log(text);
}