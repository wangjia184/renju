<script lang="ts">
	//import wasm from "../../renjuai/Cargo.toml";
	import Board from "./Board.svelte";
	import { onMount } from "svelte";
	import {
		Button,
		Modal,
		ModalBody,
		ModalHeader,
		Progress,
	} from "sveltestrap";

	onMount(async () => {});

	let isLoseOpen = false;
	let isWonOpen = false;
	let showLoading = true;
	let showStoneChoice = false;
	let loadingPercentage = 0;
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
				loadingPercentage = 100;
				setTimeout(() => {
					showLoading = false;
					showStoneChoice = true;
				}, 500);
				break;
			}

			case "progress": {
				loadingPercentage = data.progress * 100;
				break;
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
			if (boardstate && boardstate.state == "MachineThinking") {
				setTimeout(
					async () => (boardstate = await machineMove()),
					60000
				);
			}
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
		boardstate = await promise;
		if (boardstate && boardstate.state == "MachineThinking") {
			setTimeout(async () => (boardstate = await machineMove()), 1000);
		}
		showStoneChoice = false;
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

	$: boardstate,
		(() => {
			if (boardstate && boardstate.state == "HumanWon") {
				isWonOpen = true;
				setTimeout(() => (isWonOpen = false), 4000);
			} else if (boardstate && boardstate.state == "MachineWon") {
				isLoseOpen = true;
				setTimeout(() => (isLoseOpen = false), 4000);
			} else {
				// TODO : draw
			}
		})();
</script>

{#if isWonOpen || isLoseOpen}
	<div
		class="position-absolute top-0 bottom-0 start-0 end-0 d-flex align-items-center adjust-content-center justify-content-center"
		style="z-index:1000"
	>
		{#if isWonOpen}
			<img
				class="won_animated_text"
				src="/assets/youwon.png"
				alt="You won!"
			/>
		{:else if isLoseOpen}
			<img
				class="lost_animated_text"
				src="/assets/youlost.png"
				alt="You lost!"
			/>
		{/if}
	</div>
{/if}

<Board on:stone={onStone} {boardstate} />

<Modal isOpen={showLoading} backdrop="static" class="modal-dialog-centered">
	<ModalHeader>Loading ...</ModalHeader>
	<ModalBody>
		<Progress animated color="success" value={loadingPercentage} />
	</ModalBody>
</Modal>

<Modal isOpen={showStoneChoice} size="xl" class="modal-dialog-centered ">
	<ModalHeader>Choose your stone</ModalHeader>
	<ModalBody>
		<div class="d-flex flex-row mb-4">
			<div class="flex-grow-1 text-center">
				<a href="/#" on:click={() => startGame(true)}>
					<img
						src="/assets/black_basket.png"
						alt="Black"
						class="color_selector"
					/>
				</a>
			</div>
			<div class="flex-grow-1 text-center">
				<a href="/#" on:click={() => startGame(false)}>
					<img
						src="/assets/white_basket.png"
						alt="White"
						class="color_selector"
					/>
				</a>
			</div>
		</div>
	</ModalBody>
</Modal>

<style>
	.top_basket {
		width: 100%;
		height: 200px;
		background-repeat: no-repeat;
		background-size: contain;
		background-position: center top;
	}

	.bottom_basket {
		width: 100%;
		height: 200px;
		background-repeat: no-repeat;
		background-size: contain;
		background-position: center bottom;
	}

	.color_selector {
		cursor: pointer;
		opacity: 0.5;
	}

	.color_selector:hover {
		opacity: 1;
	}

	.won_animated_text {
		animation: won_frames 1s ease-in-out infinite;
	}
	@keyframes won_frames {
		0% {
			transform: scale(1);
		}
		50% {
			transform: scale(1.5);
		}
		100% {
			transform: scale(1);
		}
	}

	.lost_animated_text {
		animation: lost_frames 3s ease-in-out infinite;
	}
	@keyframes lost_frames {
		0% {
			transform: scale(1);
		}
		50% {
			transform: scale(1.2);
		}
		100% {
			transform: scale(1);
		}
	}
</style>
