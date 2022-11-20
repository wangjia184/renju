<script>
	//import wasm from "../../renjuai/Cargo.toml";
	import Board from './Board.svelte';
	import { onMount } from "svelte";
	/*
	onMount(async () => {
		const bindings = await wasm();
		var ret = bindings.parse("dd");
		console.log(ret);
	});
	*/
	const worker = new Worker("/worker.js");
	worker.onmessage = (evt) => {
		const data = evt.data;
		console.log(`Received message from worker: ${data.type} ${data.message}`)
	};
</script>

<Board />