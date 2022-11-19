
import("./ai/renjuai.js")
    .then((module) => {
        module.default()
            .then(() => {
                console.log(module.parse('x'));
            })
            .catch((err) => {
                console.log('Unable to load AI module', err);
            });
        console.log(module);
    })
    .catch((err) => {
        console.log('Unable to load renjuai.js', err);
    });