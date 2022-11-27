self.postMessage({ type: 'progress', progress: 0, message: 'Loading tensorflow' });

import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.1.0/dist/tf.min.js")
    .then(async () => {

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

        let brain = new RenjuBrain(wasm);
        brain.loop();
    })
    .catch((err) => {
        self.postMessage({ type: 'error', message: 'Unable to load tensorflow.js' });
        console.log('Unable to load tensorflow.js', err);
    });




function console_log(text) {
    console.log(text);
}


const commands = new Array();

self.onmessage = (evt) => {
    const data = evt.data;

    switch (data.type) {
        case 'start': {
            commands.push(new StartGameCommand(data.id, data.kwargs));
            break;
        }

        case 'human-move': {
            commands.push(new HumanMoveCommand(data.id, data.kwargs));
            break;
        }

        case 'machine-move': {
            commands.push(new MachineMoveCommand(data.id, data.kwargs));
            break;
        }
    }
};



class RenjuBrain {
    constructor(wasm) {
        this.wasm = wasm;
        this.brain = new wasm.Brain();
    }

    reset(kwargs) {
        return this.brain.reset(kwargs.human_play_black);
    }

    human_move(kwargs) {
        return this.brain.human_move(kwargs.row, kwargs.col);
    }

    machine_move(kwargs) {
        return this.brain.machine_move();
    }

    async loop() {
        var cmd = null;
        while ((cmd = commands.shift())) {
            this.result = await cmd.execute(this);
        }
        if (this.result && this.result.state == 'MachineThinking') {
            await this.brain.think(50);
        }
        setTimeout(() => {
            this.loop();
        }, 0);
    }

}




class CommandBase {
    constructor(id) {
        this.id = id;
    }

    reply(data) {
        self.postMessage({ type: 'reply', id: this.id, data: data });
    }
}


class StartGameCommand extends CommandBase {
    constructor(id, kwargs) {
        super(id);
        this.kwargs = kwargs;
    }

    async execute(brain/*:RenjuBrain*/) {
        const result = brain.reset(this.kwargs);
        super.reply(result);
        return result;
    }
}


class HumanMoveCommand extends CommandBase {
    constructor(id, kwargs) {
        super(id);
        this.kwargs = kwargs;
    }

    async execute(brain/*:RenjuBrain*/) {
        const result = await brain.human_move(this.kwargs);
        super.reply(result);
        return result;
    }
}


class MachineMoveCommand extends CommandBase {
    constructor(id, kwargs) {
        super(id);
        this.kwargs = kwargs;
    }

    async execute(brain/*:RenjuBrain*/) {
        const result = await brain.machine_move(this.kwargs);
        super.reply(result);
        return result;
    }
}