self.postMessage({ type: 'progress', progress: 0, message: 'Loading tensorflow' });

import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.1.0/dist/tf.min.js").then(async () => {

    // successfully loaded tensorflow.js
    self.postMessage({ type: 'progress', progress: 0.1, message: 'Loading tensorflow WASM backend' });

    try {
        let backend = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.1.0/dist/tf-backend-wasm.min.js');
        self.postMessage({ type: 'progress', progress: 0.2, message: 'Initialize tensorflow WASM backend' });

        tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.1.0/dist/');

        await tf.setBackend('wasm');
        self.postMessage({ type: 'progress', progress: 0.3, message: 'Loading model' });
    } catch (e) {
        self.postMessage({ type: 'error', message: 'Failed to load tensorflow WASM backend' });
        console.log('Failed to initialize tensorflow WASM backend', e);
        return null;
    }

    const model = await (async () => {
        try {
            return await tf.loadGraphModel('./ai/model.json');
        }
        catch (e) {
            console.log('Failed to load model', e);
            return null;
        }
    })();

    if (!model) {
        self.postMessage({ type: 'error', message: 'Failed to load model' });
        return;
    } else {
        self.postMessage({ type: 'progress', progress: 0.6, message: 'Loading WASM module' });
    }




    const wasm = await (async () => {
        try {
            return await import("./ai/renjuai.js");
        }
        catch (e) {
            console.log('Failed to load renjuai.js', e);
            return null;
        }
    })();

    if (!model) {
        self.postMessage({ type: 'error', message: 'Failed to load WASM module' });
        return;
    } else {
        self.postMessage({ type: 'progress', progress: 0.9, message: 'Initializing WASM module' });
    }


    try {
        await wasm.default();
        self.postMessage({ type: 'ready' });
    }
    catch (e) {
        console.log('Failed to initialize WASM module', e);
        self.postMessage({ type: 'error', message: 'Failed to initialize WASM module' });
        return;
    }



    self.predict = (state_tensor) => {
        const input = tf.tensor([state_tensor]);

        const tensors = model.predict(input);
        const probabilities = tensors[0].arraySync()[0];
        const score = tensors[1].dataSync();
        input.dispose();
        tensors[0].dispose();
        tensors[1].dispose();

        let prediction = new wasm.Prediction();
        prediction.score = score;
        prediction.probabilities = probabilities;

        return prediction;
    };

    wasm.test('s');


})
    .catch((err) => {
        self.postMessage({ type: 'error', message: 'Unable to load tensorflow.js' });
        console.log('Unable to load tensorflow.js', err);
    });




function console_log(text) {
    console.log(text);
}

