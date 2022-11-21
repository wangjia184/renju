<script lang="ts">
	//import wasm from "../../renjuai/Cargo.toml";
	import Board from "./Board.svelte";
	import { onMount } from "svelte";

	onMount(async () => {});

	let boardstate = null;
	let idSource = 0;
	const commandMap = {};
	const worker = new Worker("/worker.js");
	worker.onmessage = async (evt) => {
		const data = evt.data;
		switch (data.type) {
			case "reply": {
				const handler = commandMap[data.id];
				if (handler) {
					handler.resolve(data.data);
				} else {
					console.log("Unable to find the promise.", data);
				}
				break;
			}

			case "ready": {
				boardstate = await startGame(true);
			}

			default: {
				console.log(data);
				break;
			}
		}
	};

	const onStone = async (evt) => {
		if (boardstate && boardstate.state == "HumanThinking") {
			const pos = evt.detail.pos;
			boardstate = await humanMove(pos[0], pos[1]);

			setTimeout(async () => (boardstate = await machineMove()), 5000);
		}
	};

	const registerPromise = () => {
		const id = idSource++;
		const promise = new Promise((resolve, reject) => {
			commandMap[id] = {
				resolve: resolve,
				reject: reject,
			};
		});
		return { id: id, promise: promise };
	};

	const startGame = async (human_play_black) => {
		const { id, promise } = registerPromise();
		const msg = {
			type: "start",
			id: id,
			kwargs: {
				human_play_black: human_play_black,
			},
		};

		worker.postMessage(msg);
		return await promise;
	};

	const humanMove = async (row, col) => {
		const { id, promise } = registerPromise();
		const msg = {
			type: "human-move",
			id: id,
			kwargs: {
				row: row,
				col: col,
			},
		};

		worker.postMessage(msg);
		return await promise;
	};

	const machineMove = async () => {
		const { id, promise } = registerPromise();
		const msg = {
			type: "machine-move",
			id: id,
			kwargs: {},
		};

		worker.postMessage(msg);
		return await promise;
	};
</script>

<Board on:stone={onStone} {boardstate} />
